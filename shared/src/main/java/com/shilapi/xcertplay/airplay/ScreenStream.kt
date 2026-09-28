package com.shilapi.xcertplay.airplay

import android.os.SystemClock
import android.util.Log
import java.io.Closeable
import java.io.InputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

enum class VideoCodec { H264, H265 }

/**
 * Receives one CarPlay screen stream on a TCP data port.
 *
 * Each message is a 128-byte AirPlayScreenHeader followed by a body: a clear VideoConfig
 * (avcC/hvcC) or a ChaCha20-Poly1305 sealed VideoFrame. The key is the DataStream output key
 * and the per-frame nonce is an 8-byte little-endian counter.
 */
class ScreenStream(private val key: ByteArray) : Closeable {
    interface Listener {
        fun onCodec(codec: VideoCodec) {}
        fun onConfig(codecData: ByteArray) {}
        fun onFrame(naluBytes: ByteArray, timestampMs: Long) {}
        fun onClosed(cause: Throwable?) {}
    }

    private val closed = AtomicBoolean(false)
    private val frameCounter = AtomicLong(0)
    private val firstFrameLogged = AtomicBoolean(false)
    private val controlHeaderLogged = AtomicBoolean(false)
    private var lastTimestamp: Int? = null
    private var server: ServerSocket? = null
    private var socket: Socket? = null
    private var thread: Thread? = null
    @Volatile private var listener: Listener = object : Listener {}

    fun listen(listener: Listener): Int {
        this.listener = listener
        val bound = ServerSocket()
        bound.reuseAddress = true
        bound.bind(InetSocketAddress(InetAddress.getByName("::"), 0))
        server = bound
        thread = Thread({ accept(bound) }, "airplay-screen").apply { isDaemon = true; start() }
        return bound.localPort
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        safeClose(socket)
        safeClose(server)
        thread?.interrupt()
    }

    private fun accept(bound: ServerSocket) {
        try {
            val accepted = bound.accept()
            // Scene changes arrive as bursts; a small receive window makes the
            // phone stall behind TCP flow control until it looks like a freeze.
            accepted.receiveBufferSize = RECEIVE_BUFFER_BYTES
            accepted.tcpNoDelay = true
            socket = accepted
            run(accepted)
        } catch (error: Exception) {
            if (!closed.get()) listener.onClosed(error)
        }
    }

    private fun run(sock: Socket) {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY)
        var failure: Throwable? = null
        var lastMessageMs = SystemClock.elapsedRealtime()
        try {
            val input = sock.getInputStream()
            while (!closed.get()) {
                val header = readFully(input, HEADER_LEN) ?: break
                val bodySize = readU32Le(header, 0)
                if (bodySize > MAX_BODY) break
                val body = readFully(input, bodySize) ?: break
                val nowMs = SystemClock.elapsedRealtime()
                val gapMs = nowMs - lastMessageMs
                lastMessageMs = nowMs
                // A scene-change IDR stalls the stream while the phone encodes and
                // transfers it; separating that stall from receiver-side decode
                // time needs both halves logged.
                if (gapMs > STREAM_GAP_LOG_THRESHOLD_MS) {
                    Log.i(
                        TAG,
                        "video stream gap=${gapMs}ms then opcode=" +
                            "${header[OPCODE_OFFSET].toInt() and 0xff} body=${body.size} " +
                            "ts=${readU32Le(header, TIMESTAMP_OFFSET).toLong() and 0xffff_ffffL}",
                    )
                }
                onMessage(header, body)
            }
        } catch (error: Exception) {
            failure = error
        } finally {
            if (socket === sock) socket = null
            safeClose(sock)
            if (!closed.get()) listener.onClosed(failure)
        }
    }

    private fun onMessage(header: ByteArray, body: ByteArray) {
        if (header[OPCODE_OFFSET].toInt() and 0xff !in OP_VIDEO_FRAME..OP_VIDEO_CONFIG &&
            controlHeaderLogged.compareAndSet(false, true)
        ) {
            // Unknown control message: dump once to identify whether the phone
            // expects a response that would gate its send cadence.
            Log.i(
                TAG,
                "video stream control opcode=${header[OPCODE_OFFSET].toInt() and 0xff} " +
                    "header=${header.toHexString()}",
            )
        }
        when (header[OPCODE_OFFSET].toInt() and 0xff) {
            OP_VIDEO_FRAME -> {
                val payload = if (body.size >= ScreenCodec.TAG_SIZE) {
                    ScreenCodec.decryptFrame(key, frameCounter.get(), header, body)
                        .also { frameCounter.incrementAndGet() }
                } else {
                    body
                }
                logFrameTimestamp(header)
                if (firstFrameLogged.compareAndSet(false, true)) {
                    Log.i(
                        TAG,
                        "video first decrypted frame sealed=${body.size} plain=${payload.size} " +
                        "head=${payload.hexPrefix(16)}",
                    )
                }
                listener.onFrame(
                    ScreenCodec.lengthPrefixedToAnnexB(payload),
                    readU32Le(header, TIMESTAMP_OFFSET).toLong() and 0xffff_ffffL,
                )
            }
            OP_VIDEO_CONFIG -> {
                val (codec, codecData) = ScreenCodec.detectConfig(body)
                Log.i(TAG, "video codec config codec=$codec body=${body.size} data=${codecData.size}")
                listener.onCodec(codec)
                listener.onConfig(codecData)
            }
        }
    }

    /**
     * Logs the candidate presentation timestamp in the 128-byte header (LE u32 at
     * offset 8) so its units and cadence can be verified before render pacing
     * uses it. Sparse: the first frames, then every [TIMESTAMP_LOG_INTERVAL]-th.
     */
    private fun logFrameTimestamp(header: ByteArray) {
        val sequence = frameCounter.get()
        if (sequence >= TIMESTAMP_LOG_SKIP && sequence % TIMESTAMP_LOG_INTERVAL != 0L) return
        val raw = readU32Le(header, TIMESTAMP_OFFSET)
        val last = lastTimestamp
        lastTimestamp = raw
        if (sequence < TIMESTAMP_LOG_SKIP || last == null) {
            Log.i(TAG, "video frame ts raw=$raw seq=$sequence")
            return
        }
        val delta = (raw - last).toInt()
        Log.i(TAG, "video frame ts raw=$raw delta=$delta seq=$sequence")
    }

    private fun readFully(input: InputStream, length: Int): ByteArray? {
        if (length < 0) return null
        val output = ByteArray(length)
        var offset = 0
        while (offset < length) {
            val read = input.read(output, offset, length - offset)
            if (read < 0) return null
            offset += read
        }
        return output
    }

    private companion object {
        const val TAG = "xcertplay-usb"
        const val HEADER_LEN = 128
        const val STREAM_GAP_LOG_THRESHOLD_MS = 150L
        const val RECEIVE_BUFFER_BYTES = 2 * 1024 * 1024
        const val OPCODE_OFFSET = 4
        const val TIMESTAMP_OFFSET = 8
        const val TIMESTAMP_LOG_SKIP = 3L
        const val TIMESTAMP_LOG_INTERVAL = 300L
        const val OP_VIDEO_FRAME = 0
        const val OP_VIDEO_CONFIG = 1
        const val MAX_BODY = 8 * 1024 * 1024
    }
}

private fun ByteArray.hexPrefix(length: Int): String =
    take(length).joinToString("") { "%02x".format(it.toInt() and 0xff) }

/** Extracts the avcC/hvcC codec-data record from a VideoConfig payload. */
object ScreenCodec {
    fun decryptFrame(key: ByteArray, counter: Long, header: ByteArray, body: ByteArray): ByteArray =
        if (body.size < TAG_SIZE) body
        else AirPlayCrypto.chachaOpen(key, AirPlayCrypto.nonce64(counter), body, header)

    /**
     * Replaces each four-byte NAL length with an Annex B start code in place.
     *
     * The payload is left untouched unless every length-prefixed NAL is valid, so malformed
     * input keeps its original bytes for the normal decoder error path.
     */
    fun lengthPrefixedToAnnexB(payload: ByteArray): ByteArray {
        if (payload.size < 4 || payload.startsWithStartCode()) return payload

        var offset = 0
        while (offset + 4 <= payload.size) {
            val length = readU32Be(payload, offset)
            offset += 4
            if (length <= 0 || offset + length > payload.size) return payload
            offset += length
        }
        if (offset != payload.size) return payload

        offset = 0
        while (offset + 4 <= payload.size) {
            val length = readU32Be(payload, offset)
            payload[offset] = 0
            payload[offset + 1] = 0
            payload[offset + 2] = 0
            payload[offset + 3] = 1
            offset += 4 + length
        }
        return payload
    }

    fun detectConfig(payload: ByteArray): Pair<VideoCodec, ByteArray> {
        for (index in 4..payload.size - 4) {
            val fourcc = String(payload, index, 4, Charsets.US_ASCII)
            when (fourcc) {
                "hvcC" -> return VideoCodec.H265 to payload.copyOfRange(index + 4, payload.size)
                "avcC" -> return VideoCodec.H264 to payload.copyOfRange(index + 4, payload.size)
            }
        }
        return if (looksLikeAvcC(payload)) VideoCodec.H264 to payload else VideoCodec.H265 to payload
    }

    private fun looksLikeAvcC(payload: ByteArray): Boolean {
        if (payload.size < 9) return false
        if ((payload[5].toInt() and 0x1f) < 1) return false
        val spsLength = readU16Be(payload, 6)
        if (8 + spsLength > payload.size) return false
        return (payload[8].toInt() and 0x1f) == 7
    }

    private fun readU16Be(source: ByteArray, offset: Int): Int =
        ((source[offset].toInt() and 0xff) shl 8) or (source[offset + 1].toInt() and 0xff)

    const val TAG_SIZE = 16
}

private fun ByteArray.startsWithStartCode(): Boolean =
    size >= 4 &&
        this[0] == 0.toByte() &&
        this[1] == 0.toByte() &&
        this[2] == 0.toByte() &&
        this[3] == 1.toByte()

private fun readU32Be(source: ByteArray, offset: Int): Int =
    ((source[offset].toInt() and 0xff) shl 24) or
        ((source[offset + 1].toInt() and 0xff) shl 16) or
        ((source[offset + 2].toInt() and 0xff) shl 8) or
        (source[offset + 3].toInt() and 0xff)

private fun readU32Le(source: ByteArray, offset: Int): Int =
    (source[offset].toInt() and 0xff) or
        ((source[offset + 1].toInt() and 0xff) shl 8) or
        ((source[offset + 2].toInt() and 0xff) shl 16) or
        ((source[offset + 3].toInt() and 0xff) shl 24)
