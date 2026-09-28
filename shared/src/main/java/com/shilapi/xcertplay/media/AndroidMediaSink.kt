package com.shilapi.xcertplay.media

import android.media.AudioAttributes
import android.media.AudioFormat as AndroidAudioFormat
import android.media.AudioTrack
import android.media.MediaCodec
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Build
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import com.shilapi.xcertplay.airplay.AudioCodecKind
import com.shilapi.xcertplay.airplay.AudioFormat
import com.shilapi.xcertplay.airplay.MediaSink
import com.shilapi.xcertplay.airplay.MicrophoneConfig
import com.shilapi.xcertplay.airplay.VideoCodec
import com.shilapi.xcertplay.airplay.toHexString
import java.io.Closeable
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.LinkedBlockingQueue

/**
 * Android rendering backend for the CarPlay media engine. Video frames are
 * decoded with MediaCodec onto a Surface; audio streams are decoded to PCM and
 * played through AudioTrack. Call [close] when the session tears down.
 */
class AndroidMediaSink(
    surface: Surface? = null,
    private val videoWidth: Int = 1280,
    private val videoHeight: Int = 720,
    private val preferSoftwareHevcDecoder: Boolean = false,
    private val advancedAudioChannelMapping: Boolean = false,
    onScreenStreamActiveChanged: ((Int, Boolean) -> Unit)? = null,
    private val onDiagnostic: ((String) -> Unit)? = null,
    private val onKeyframeRequest: (() -> Unit)? = null,
) : MediaSink {
    private val defaultSurface = surface
    @Volatile private var screenStreamActiveChanged = onScreenStreamActiveChanged
    private val surfaces = ConcurrentHashMap<Int, Surface>()
    private val videoDecoders = ConcurrentHashMap<Int, VideoDecoder>()
    private val audioRenderers = ConcurrentHashMap<Int, AudioRenderer>()
    private val microphoneUplinks = ConcurrentHashMap<Int, MicrophoneUplink>()
    private val pendingVideoCodec = ConcurrentHashMap<Int, VideoCodec>()

    fun setSurface(type: Int, surface: Surface) {
        surfaces[type] = surface
        videoDecoders[type]?.setSurface(surface)
    }

    fun clearSurface(type: Int, surface: Surface) {
        if (surfaces.remove(type, surface)) videoDecoders[type]?.setSurface(null)
    }

    fun setScreenStreamActiveChangedListener(listener: ((Int, Boolean) -> Unit)?) {
        screenStreamActiveChanged = listener
    }

    override fun onVideoCodec(type: Int, codec: VideoCodec) {
        pendingVideoCodec[type] = codec
    }

    override fun onVideoConfig(type: Int, codecData: ByteArray) {
        val codec = pendingVideoCodec[type] ?: VideoCodec.H264
        videoDecoder(type).configure(codec, codecData)
    }

    override fun onVideoFrame(type: Int, naluBytes: ByteArray, timestampMs: Long) {
        videoDecoder(type).submit(naluBytes, timestampMs)
    }

    override fun onScreenStreamActive(type: Int, active: Boolean) {
        screenStreamActiveChanged?.invoke(type, active)
    }

    override fun onAudioStarted(type: Int, format: AudioFormat, firstSample: Int) {
        audioRenderer(type, format).start()
    }

    override fun onAudioRtp(type: Int, format: AudioFormat, rtp: ByteArray, sample: Int) {
        audioRenderer(type, format).submit(rtp, sample)
    }

    override fun onAudioStopped(type: Int) {
        audioRenderers.remove(type)?.close()
    }

    override fun onMicrophoneStarted(type: Int, config: MicrophoneConfig) {
        val uplink = microphoneUplinks.computeIfAbsent(type) { MicrophoneUplink(config) }
        if (!uplink.start()) microphoneUplinks.remove(type, uplink)
    }

    override fun onMicrophoneStopped(type: Int) {
        microphoneUplinks.remove(type)?.close()
    }

    fun close() {
        videoDecoders.values.forEach(VideoDecoder::close)
        videoDecoders.clear()
        audioRenderers.values.forEach(AudioRenderer::close)
        audioRenderers.clear()
        microphoneUplinks.values.forEach(MicrophoneUplink::close)
        microphoneUplinks.clear()
    }

    private fun videoDecoder(type: Int): VideoDecoder =
        videoDecoders.computeIfAbsent(type) {
            VideoDecoder(
                surfaces[type] ?: defaultSurface,
                videoWidth,
                videoHeight,
                preferSoftwareHevcDecoder,
                onDiagnostic,
                onKeyframeRequest,
            )
        }

    @Synchronized
    private fun audioRenderer(type: Int, format: AudioFormat): AudioRenderer {
        val existing = audioRenderers[type]
        if (existing?.format == format) return existing
        existing?.close()
        return AudioRenderer(format, advancedAudioChannelMapping).also { audioRenderers[type] = it }
    }
}

private sealed interface VideoJob {
    data class Config(val codec: VideoCodec, val codecData: ByteArray) : VideoJob
    data class Frame(val nalus: ByteArray, val timestampMs: Long) : VideoJob
    data class SurfaceChanged(val surface: Surface?) : VideoJob
}

/** Serial MediaCodec video decoder: one worker owns configure and frame feeding. */
private class VideoDecoder(
    surface: Surface?,
    private val width: Int,
    private val height: Int,
    private val preferSoftwareHevcDecoder: Boolean,
    private val onDiagnostic: ((String) -> Unit)? = null,
    private val onKeyframeRequest: (() -> Unit)? = null,
) : Closeable {
    private val queue = LinkedBlockingQueue<VideoJob>()
    @Volatile private var running = true
    @Volatile private var decoder: MediaCodec? = null
    private var outputSurface: Surface? = surface
    private var lastConfig: VideoJob.Config? = null

    /**
     * True while the decoder has no usable reference picture and input must be
     * shed until the next IDR/IRAP frame. Dropping a mid-GOP frame silently
     * would corrupt every frame after it until the next keyframe.
     */
    @Volatile private var awaitingKeyframe = false
    private var backlogSinceMs = 0L
    private var renderedFrameLogged = false
    private var submittedFrameLogged = false
    private var duplicateConfigLogged = false
    private val probeTimestampUnits = ArrayList<Long>(PACE_PROBE_FRAMES)
    private val probeArrivalNs = ArrayList<Long>(PACE_PROBE_FRAMES)
    private var pacingState = PACING_PROBING
    private var unitsPerMs = 1.0
    private var anchorArrivalNs = 0L
    private var anchorTimestampUnits = 0L
    private var lastTimestampUnits = 0L
    private var lastFrameArrivalNs = 0L
    private var lastDiagMs = 0L
    private var awaitingKeyframeSinceMs = 0L
    private var lastFeedMs = 0L
    private var lastKeyframeRequestMs = 0L
    private val thread = Thread(::run, "carplay-video").apply { isDaemon = true; start() }

    fun configure(codec: VideoCodec, codecData: ByteArray) {
        queue.offer(VideoJob.Config(codec, codecData))
    }

    fun submit(nalus: ByteArray, timestampMs: Long) {
        queue.offer(VideoJob.Frame(nalus, timestampMs))
    }

    fun setSurface(surface: Surface?) {
        queue.offer(VideoJob.SurfaceChanged(surface))
    }

    override fun close() {
        running = false
        thread.interrupt()
    }

    private fun run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY)
        try {
            while (running) {
                val job = queue.take()
                try {
                    when (job) {
                        is VideoJob.Config -> configureDecoder(job)
                        is VideoJob.Frame -> handleFrame(job)
                        is VideoJob.SurfaceChanged -> changeSurface(job.surface)
                    }
                } catch (error: Exception) {
                    if (running) Log.e(TAG, "video decoder job failed: ${job.javaClass.simpleName}", error)
                    releaseDecoder()
                }
            }
        } catch (_: InterruptedException) {
            // Worker shut down.
        } finally {
            releaseDecoder()
        }
    }

    private fun configureDecoder(config: VideoJob.Config) {
        val previous = lastConfig
        if (
            decoder != null &&
            previous?.codec == config.codec &&
            previous.codecData.contentEquals(config.codecData)
        ) {
            if (!duplicateConfigLogged) {
                duplicateConfigLogged = true
                Log.i(TAG, "video decoder config unchanged; keeping existing decoder")
            }
            return
        }
        lastConfig = config
        duplicateConfigLogged = false
        awaitingKeyframe = false
        backlogSinceMs = 0L
        // A reconfigured stream restarts its timestamp base; re-probe pacing.
        probeTimestampUnits.clear()
        probeArrivalNs.clear()
        pacingState = PACING_PROBING
        releaseDecoder()
        val surface = outputSurface ?: return
        val codec = config.codec
        val codecData = config.codecData
        val mime = if (codec == VideoCodec.H265) MediaFormat.MIMETYPE_VIDEO_HEVC
        else MediaFormat.MIMETYPE_VIDEO_AVC
        val format = MediaFormat.createVideoFormat(mime, width, height).apply {
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, MAX_INPUT_SIZE)
        }
        if (codec == VideoCodec.H265) {
            val csd = MediaCodecSupport.hevcCodecSpecificData(codecData)
            if (csd.isNotEmpty()) format.setByteBuffer("csd-0", ByteBuffer.wrap(csd))
        } else {
            val (sps, pps) = MediaCodecSupport.avcParameterSets(codecData)
            if (sps.isNotEmpty()) format.setByteBuffer("csd-0", ByteBuffer.wrap(START_CODE + sps))
            if (pps.isNotEmpty()) format.setByteBuffer("csd-1", ByteBuffer.wrap(START_CODE + pps))
        }
        val next = try {
            createDecoder(mime).also {
                it.configure(format, surface, null, 0)
                it.start()
            }
        } catch (error: Exception) {
            Log.e(TAG, "video decoder configure failed mime=$mime size=${width}x$height", error)
            null
        }
        decoder = next
        renderedFrameLogged = false
        submittedFrameLogged = false
        if (next != null) {
            Log.i(
                TAG,
                "video decoder configured name=${next.name} mime=$mime size=${width}x$height",
            )
        }
    }

    private fun createDecoder(mime: String): MediaCodec {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
            mime == MediaFormat.MIMETYPE_VIDEO_HEVC &&
            preferSoftwareHevcDecoder
        ) {
            val software = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.firstOrNull {
                !it.isEncoder && it.isSoftwareOnly && mime in it.supportedTypes
            }
            if (software != null) {
                try {
                    return MediaCodec.createByCodecName(software.name)
                } catch (error: Exception) {
                    Log.w(TAG, "software HEVC decoder unavailable name=${software.name}", error)
                }
            }
        }
        return MediaCodec.createDecoderByType(mime)
    }

    private fun changeSurface(surface: Surface?) {
        if (outputSurface === surface) return
        outputSurface = surface
        if (surface == null) {
            releaseDecoder()
            Log.i(TAG, "video decoder detached from surface")
            return
        }
        val codec = decoder
        if (codec != null) {
            try {
                codec.setOutputSurface(surface)
                Log.i(TAG, "video decoder output surface updated")
                return
            } catch (error: Exception) {
                Log.w(TAG, "video decoder output surface update failed; reconfiguring", error)
            }
        }
        releaseDecoder()
        lastConfig?.let(::configureDecoder)
    }

    /** Holds [frame] until its presentation time so Wi-Fi bursts render evenly. */
    private fun handleFrame(frame: VideoJob.Frame) {
        expireKeyframeWait()
        pace(frame.timestampMs)
        detectStarvation()
        if (enforceBacklogCap() && !isKeyframe(frame.nalus)) return
        feed(frame.nalus)
    }

    /**
     * A mid-stream frame arriving after a long dry spell means the phone's sender
     * stalled (scene change bursts do this). Asking it for a keyframe makes it
     * cut the stale GOP and resume full-rate output instead of trickling frames
     * that reference pictures we may have shed. Static screens keep a ~1s
     * micro-frame cadence, so the threshold only trips on real stalls.
     */
    private fun detectStarvation() {
        val now = SystemClock.elapsedRealtime()
        val last = lastFeedMs
        if (last <= 0L || now - last <= KEYFRAME_STALL_THRESHOLD_MS) return
        if (now - lastKeyframeRequestMs < KEYFRAME_REQUEST_DEBOUNCE_MS) return
        lastKeyframeRequestMs = now
        diag("video stall ${now - last}ms; requesting keyframe", minIntervalMs = 5_000L)
        onKeyframeRequest?.invoke()
    }

    /**
     * CarPlay only sends keyframes on scene changes, so a resync can wait
     * forever while the phone keeps encoding P-frames. After the timeout the
     * decoder resumes mid-GOP: briefly corrupt beats permanently frozen.
     */
    private fun expireKeyframeWait() {
        if (!awaitingKeyframe) return
        val waited = SystemClock.elapsedRealtime() - awaitingKeyframeSinceMs
        if (waited <= KEYFRAME_WAIT_TIMEOUT_MS) return
        awaitingKeyframe = false
        backlogSinceMs = 0L
        diag("video resynced by timeout after ${waited}ms without a keyframe; accepting corruption", minIntervalMs = 5_000L)
    }

    /**
     * Render pacing: frames arrive over TCP in bursts, so feeding the decoder on
     * arrival makes display cadence follow network jitter. The per-frame
     * timestamp (unsigned ms) is mapped onto the local clock and the frame waits
     * for its slot. The mapping is trusted only after [PACE_PROBE_FRAMES]
     * monotonic samples; otherwise pacing is disabled and frames decode on
     * arrival as before. Cap-based re-anchoring bounds both the added lead and
     * the catch-up after stalls.
     */
    private fun pace(timestampUnits: Long) {
        if (timestampUnits < 0) return
        val nowNs = System.nanoTime()
        if (pacingState == PACING_DISABLED) return
        if (pacingState == PACING_PROBING) {
            probeTimestampUnits.add(timestampUnits)
            probeArrivalNs.add(nowNs)
            lastFrameArrivalNs = nowNs
            if (probeTimestampUnits.size < PACE_PROBE_FRAMES) return
            val calibrated = calibrate()
            if (calibrated == null) {
                pacingState = PACING_DISABLED
                diag("video render pacing disabled; probe timestamps=$probeTimestampUnits")
                return
            }
            unitsPerMs = calibrated
            pacingState = PACING_ACTIVE
            anchorArrivalNs = probeArrivalNs.last()
            anchorTimestampUnits = probeTimestampUnits.last()
            lastTimestampUnits = anchorTimestampUnits
            diag("video render pacing enabled unitsPerMs=$calibrated")
            return
        }
        val gapNs = nowNs - lastFrameArrivalNs
        lastFrameArrivalNs = nowNs
        if (gapNs > PACE_IDLE_GAP_NS) {
            // First frame after idle is usually the direct reaction to a touch;
            // render it at arrival, then resume pacing with the standard lead.
            // Anchoring at now + LEAD (not bare now) keeps the mapping latency
            // positive so following frames still get spread instead of all
            // collapsing to render-on-arrival.
            anchorArrivalNs = nowNs + PACE_MAX_LEAD_NS
            anchorTimestampUnits = timestampUnits
            lastTimestampUnits = timestampUnits
            return
        }
        val stepFromLastMs = deltaMs(lastTimestampUnits, timestampUnits)
        refineRate(lastTimestampUnits, timestampUnits, gapNs / 1_000_000.0)
        lastTimestampUnits = timestampUnits
        if (stepFromLastMs > PACE_REANCHOR_STEP_MS) {
            // Stream restart or clock jump; keep rendering instead of sleeping it off.
            anchorArrivalNs = nowNs + PACE_MAX_LEAD_NS
            anchorTimestampUnits = timestampUnits
            diag("video pacing re-anchored after jump stepMs=$stepFromLastMs", minIntervalMs = 10_000L)
            return
        }
        var targetNs = anchorArrivalNs + deltaMs(anchorTimestampUnits, timestampUnits) * 1_000_000L
        if (targetNs > nowNs + PACE_MAX_LEAD_NS) {
            anchorArrivalNs = nowNs + PACE_MAX_LEAD_NS
            anchorTimestampUnits = timestampUnits
            targetNs = anchorArrivalNs
        } else if (targetNs < nowNs - PACE_MAX_LAG_NS) {
            // Far behind after a stall; decode now and resume pacing from this frame.
            anchorArrivalNs = nowNs + PACE_MAX_LEAD_NS
            anchorTimestampUnits = timestampUnits
            diag(
                "video pacing re-anchored after lag unitsPerMs=$unitsPerMs",
                minIntervalMs = 10_000L,
            )
            return
        }
        // A mis-calibrated rate can pin every frame at the full lead sleep; with
        // frames queuing behind, pacing must yield so it can never cause the
        // backlog that triggers a resync.
        if (queue.size <= PACE_SLEEP_MAX_QUEUE) sleepUntil(targetNs)
    }

    /**
     * Keeps the calibrated clock rate honest: a single probe window carries the
     * Wi-Fi jitter of those five frames (observed ~10% on one car), which then
     * drifts the mapping into a re-anchor every few seconds. Each frame with a
     * plausible inter-arrival gap nudges the rate toward the observed one, so
     * the drift converges to zero instead of accumulating.
     */
    private fun refineRate(previousUnits: Long, currentUnits: Long, gapMs: Double) {
        if (gapMs !in PACE_RATE_MIN_GAP_MS..PACE_RATE_MAX_GAP_MS) return
        val stepMs = unsignedDelta(previousUnits, currentUnits) / unitsPerMs
        if (stepMs !in PACE_MIN_FRAME_MS..PACE_RATE_MAX_FRAME_MS) return
        val observed = unsignedDelta(previousUnits, currentUnits) / gapMs
        if (!observed.isFinite() || observed <= 0.0) return
        if (Math.abs(observed - unitsPerMs) > unitsPerMs * 0.5) return
        unitsPerMs += (observed - unitsPerMs) * PACE_RATE_SMOOTHING
    }

    /**
     * Derives the stream clock rate (units per wall millisecond) from the probe
     * window. The header counter's unit is not fixed across phones — one iPhone
     * ticked at 2^31 Hz, wrapping every two seconds — so the rate is measured,
     * not assumed. Calibration fails when the probe window is too short (one
     * Wi-Fi burst) or the implied frame periods are out of codec range.
     */
    private fun calibrate(): Double? {
        val arrivalSpanMs = (probeArrivalNs.last() - probeArrivalNs.first()) / 1_000_000.0
        if (arrivalSpanMs < PACE_PROBE_MIN_SPAN_MS) return null
        val rate = unsignedDelta(probeTimestampUnits.first(), probeTimestampUnits.last()) / arrivalSpanMs
        if (rate <= 0.0 || !rate.isFinite()) return null
        val framePeriodsOk = probeTimestampUnits.zipWithNext { previous, next ->
            unsignedDelta(previous, next) / rate
        }.all { it in PACE_MIN_FRAME_MS..PACE_PROBE_MAX_STEP_MS }
        if (!framePeriodsOk) return null
        return rate
    }

    private fun deltaMs(fromUnits: Long, toUnits: Long): Long =
        Math.round(unsignedDelta(fromUnits, toUnits) / unitsPerMs)

    private fun unsignedDelta(from: Long, to: Long): Long = (to - from) and 0xffff_ffffL

    private fun sleepUntil(targetNs: Long) {
        while (running) {
            val remainingMs = (targetNs - System.nanoTime()) / 1_000_000L
            if (remainingMs <= 0) return
            try {
                Thread.sleep(minOf(remainingMs, PACE_SLEEP_SLICE_MS))
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            }
        }
    }

    /**
     * Mirrors a decoder event into logcat and, when wired, the exportable file
     * log so car-side diagnosis no longer requires adb. Rate-limited per call
     * site because idle re-anchors and resyncs can repeat for every touch.
     */
    private fun diag(message: String, minIntervalMs: Long = 0L) {
        Log.i(TAG, message)
        val now = SystemClock.elapsedRealtime()
        if (now - lastDiagMs < minIntervalMs) return
        lastDiagMs = now
        onDiagnostic?.invoke(message)
    }

    private fun unsignedDeltaMs(from: Long, to: Long): Long = (to - from) and 0xffff_ffffL

    /**
     * A short burst may exceed the cap while the decoder catches up, so the cap
     * must hold for [BACKLOG_SUSTAIN_MS] before shedding. Otherwise a burst that
     * the decoder can still absorb would trigger a needless resync.
     */
    private fun enforceBacklogCap(): Boolean {
        if (queue.size < LATENCY_CAP_FRAMES) {
            backlogSinceMs = 0L
            return false
        }
        val now = SystemClock.elapsedRealtime()
        if (backlogSinceMs == 0L) {
            backlogSinceMs = now
            return false
        }
        if (now - backlogSinceMs <= BACKLOG_SUSTAIN_MS) return false
        backlogSinceMs = 0L
        if (!awaitingKeyframe) beginKeyframeRecovery("backlog over $LATENCY_CAP_FRAMES frames")
        return true
    }

    private fun feed(nalus: ByteArray) {
        lastFeedMs = SystemClock.elapsedRealtime()
        val codec = decoder ?: return
        if (awaitingKeyframe && !isKeyframe(nalus)) {
            // Frames referencing the dropped picture cannot decode cleanly.
            return
        }
        val annexB = MediaCodecSupport.toAnnexB(nalus)
        if (!submittedFrameLogged) {
            submittedFrameLogged = true
            Log.i(
                TAG,
                "video decoder first input avcc=${nalus.size} annexB=${annexB.size} " +
                    "head=${annexB.take(16).joinToString("") { "%02x".format(it.toInt() and 0xff) }}",
            )
        }
        if (annexB.isEmpty()) return
        if (awaitingKeyframe) {
            awaitingKeyframe = false
            backlogSinceMs = 0L
            diag("video resynced at keyframe")
        }
        val index = awaitInputBuffer(codec)
        if (index < 0) {
            beginKeyframeRecovery("decoder input unavailable for ${INPUT_WAIT_TIMEOUT_US / 1000} ms")
            return
        }
        val input = codec.getInputBuffer(index) ?: return
        input.clear()
        if (annexB.size <= input.remaining()) {
            input.put(annexB)
            codec.queueInputBuffer(index, 0, annexB.size, System.nanoTime() / 1000, 0)
        } else {
            beginKeyframeRecovery("frame ${annexB.size} bytes exceeds decoder input buffer")
        }
        val decodeStartMs = SystemClock.elapsedRealtime()
        drainOutput(codec)
        val decodeMs = SystemClock.elapsedRealtime() - decodeStartMs
        if (decodeMs > SLOW_DECODE_LOG_MS) {
            // A scene-change IDR can tie up a weak hardware decoder; this pins
            // receiver-side decode cost against stream-side gaps.
            Log.i(TAG, "video decode slow=${decodeMs}ms bytes=${annexB.size}")
        }
    }

    /**
     * Waits up to [INPUT_WAIT_TIMEOUT_US] for an input buffer so a brief decoder
     * stall does not drop a frame and corrupt the rest of the GOP.
     */
    private fun awaitInputBuffer(codec: MediaCodec): Int {
        val deadline = System.nanoTime() + INPUT_WAIT_TIMEOUT_US * 1_000
        while (running) {
            val index = codec.dequeueInputBuffer(INPUT_TIMEOUT_US)
            if (index >= 0) return index
            if (System.nanoTime() >= deadline) return -1
        }
        return -1
    }

    /** Flushes the decoder and sheds input until the next IDR/IRAP frame. */
    private fun beginKeyframeRecovery(reason: String) {
        if (!awaitingKeyframe) awaitingKeyframeSinceMs = SystemClock.elapsedRealtime()
        awaitingKeyframe = true
        backlogSinceMs = 0L
        val now = SystemClock.elapsedRealtime()
        if (now - lastKeyframeRequestMs >= KEYFRAME_REQUEST_DEBOUNCE_MS) {
            lastKeyframeRequestMs = now
            onKeyframeRequest?.invoke()
        }
        val codec = decoder
        if (codec != null) {
            try {
                codec.flush()
            } catch (error: Exception) {
                Log.w(TAG, "video decoder flush failed during resync", error)
            }
        }
        diag("video resync at next keyframe: $reason", minIntervalMs = 2_000L)
    }

    /** Detects an IDR (H.264) or IRAP (H.265) access unit in a length-prefixed payload. */
    private fun isKeyframe(nalus: ByteArray): Boolean {
        val codec = lastConfig?.codec ?: VideoCodec.H264
        var offset = 0
        while (offset + NAL_LENGTH_PREFIX_SIZE <= nalus.size) {
            val length = ((nalus[offset].toInt() and 0xff) shl 24) or
                ((nalus[offset + 1].toInt() and 0xff) shl 16) or
                ((nalus[offset + 2].toInt() and 0xff) shl 8) or
                (nalus[offset + 3].toInt() and 0xff)
            if (length <= 0 || offset + NAL_LENGTH_PREFIX_SIZE + length > nalus.size) return false
            val header = nalus[offset + NAL_LENGTH_PREFIX_SIZE].toInt() and 0xff
            val key = when (codec) {
                VideoCodec.H264 -> (header and 0x1f) == H264_NAL_IDR
                VideoCodec.H265 -> ((header shr 1) and 0x3f) in HEVC_IRAP_MIN..HEVC_IRAP_MAX
            }
            if (key) return true
            offset += NAL_LENGTH_PREFIX_SIZE + length
        }
        return false
    }

    private fun drainOutput(codec: MediaCodec) {
        val info = MediaCodec.BufferInfo()
        while (running) {
            val index = codec.dequeueOutputBuffer(info, 0)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> return
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> logOutputFormat(codec.outputFormat)
                index >= 0 -> {
                    val render = outputSurface != null
                    codec.releaseOutputBuffer(index, render)
                    if (render && !renderedFrameLogged) {
                        renderedFrameLogged = true
                        Log.i(TAG, "video decoder rendered first frame bytes=${info.size}")
                    }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
                else -> return
            }
        }
    }

    private fun logOutputFormat(format: MediaFormat) {
        Log.i(
            TAG,
            "video decoder output format " +
                "size=${format.intOrNull(MediaFormat.KEY_WIDTH)}x" +
                "${format.intOrNull(MediaFormat.KEY_HEIGHT)} " +
                "stride=${format.intOrNull(MediaFormat.KEY_STRIDE)} " +
                "slice=${format.intOrNull(MediaFormat.KEY_SLICE_HEIGHT)} " +
                "standard=${format.intOrNull(MediaFormat.KEY_COLOR_STANDARD)} " +
                "range=${format.intOrNull(MediaFormat.KEY_COLOR_RANGE)} " +
                "transfer=${format.intOrNull(MediaFormat.KEY_COLOR_TRANSFER)}",
        )
    }

    @Synchronized
    private fun releaseDecoder() {
        val codec = decoder
        decoder = null
        if (codec != null) {
            try {
                codec.stop()
            } catch (_: Exception) {
                // Best effort.
            }
            try {
                codec.release()
            } catch (_: Exception) {
                // Best effort.
            }
        }
    }

    private companion object {
        const val TAG = "xcertplay-usb"
        const val MAX_INPUT_SIZE = 8 * 1024 * 1024
        const val INPUT_TIMEOUT_US = 10_000L
        // A scene-change IDR can occupy the hardware decoder for well over 200ms
        // on weak SoCs; giving up then would flush that very keyframe and freeze
        // the picture until the next one (a full GOP, 1-2s).
        const val INPUT_WAIT_TIMEOUT_US = 600_000L
        // Tolerating a deeper, longer backlog avoids a freeze-inducing resync on
        // short Wi-Fi bursts; the cap still bounds worst-case latency once held.
        const val LATENCY_CAP_FRAMES = 16
        const val BACKLOG_SUSTAIN_MS = 800L
        const val PACING_PROBING = 0
        const val PACING_ACTIVE = 1
        const val PACING_DISABLED = 2
        const val PACE_PROBE_FRAMES = 5
        const val PACE_PROBE_MIN_SPAN_MS = 40.0
        const val PACE_MIN_FRAME_MS = 5.0
        const val PACE_PROBE_MAX_STEP_MS = 2_000.0
        const val PACE_RATE_MIN_GAP_MS = 5.0
        const val PACE_RATE_MAX_GAP_MS = 100.0
        const val PACE_RATE_MAX_FRAME_MS = 200.0
        const val PACE_RATE_SMOOTHING = 0.05
        const val PACE_SLEEP_MAX_QUEUE = 4
        const val KEYFRAME_WAIT_TIMEOUT_MS = 2_500L
        const val KEYFRAME_STALL_THRESHOLD_MS = 400L
        const val KEYFRAME_REQUEST_DEBOUNCE_MS = 1_500L
        const val SLOW_DECODE_LOG_MS = 250L
        const val PACE_REANCHOR_STEP_MS = 10_000L
        // Lead costs touch latency directly, so it stays just deep enough to
        // absorb a multi-frame Wi-Fi burst at 30 fps.
        const val PACE_MAX_LEAD_NS = 40_000_000L
        const val PACE_MAX_LAG_NS = 1_000_000_000L
        const val PACE_IDLE_GAP_NS = 250_000_000L
        const val PACE_SLEEP_SLICE_MS = 5L
        const val NAL_LENGTH_PREFIX_SIZE = 4
        const val H264_NAL_IDR = 5
        const val HEVC_IRAP_MIN = 16
        const val HEVC_IRAP_MAX = 23
        val START_CODE = byteArrayOf(0x00, 0x00, 0x00, 0x01)
    }
}

private fun MediaFormat.intOrNull(key: String): Int? =
    if (!containsKey(key)) {
        null
    } else {
        try {
            getInteger(key)
        } catch (_: Exception) {
            null
        }
    }

/** Decodes AAC-LC/Opus to PCM and plays it, or plays wired LPCM directly. */
private class AudioRenderer(
    val format: AudioFormat,
    private val advancedAudioChannelMapping: Boolean,
) : Closeable {
    private data class AudioPacket(val rtp: ByteArray, val sample: Int)

    private val queue = LinkedBlockingQueue<AudioPacket>(MAX_QUEUED_PACKETS)
    @Volatile private var running = true
    @Volatile private var started = false
    private var codec: MediaCodec? = null
    private var track: AudioTrack? = null
    private var pcm = ByteArray(64 * 1024)
    private var playbackStarted = false
    private var prebufferBytes = 0
    private var startThresholdBytes = 0
    private var fadeApplied = false
    private var droppedPacketsLogged = false
    private var firstAacPayloadLogged = false
    private var firstOpusShortPacketLogged = false
    private var firstInputQueuedLogged = false
    private var inputQueued = 0
    private var inputDropped = 0
    private var outputBuffers = 0
    private var firstPcmLogged = false
    private val thread = Thread(::run, "carplay-audio").apply { isDaemon = true }

    fun start() {
        if (started) return
        started = true
        thread.start()
    }

    fun submit(rtp: ByteArray, sample: Int) {
        if (!started) return
        val packet = AudioPacket(rtp, sample)
        if (queue.offer(packet)) return
        // Shed the oldest packet so the freshest audio keeps playing and
        // latency stays bounded.
        queue.poll()
        if (!queue.offer(packet) && !droppedPacketsLogged) {
            droppedPacketsLogged = true
            Log.w(TAG, "audio queue full; shedding oldest packets to bound latency")
        }
    }

    override fun close() {
        running = false
        thread.interrupt()
    }

    private fun run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_AUDIO)
        try {
            when (format.codec) {
                AudioCodecKind.AAC_LC -> configureCodec(MediaFormat.MIMETYPE_AUDIO_AAC)
                AudioCodecKind.OPUS -> configureCodec(MediaFormat.MIMETYPE_AUDIO_OPUS)
                AudioCodecKind.LPCM -> Unit
            }
            createTrack()
            while (running) handle(queue.take())
        } catch (_: InterruptedException) {
            // Worker shut down.
        } catch (error: Exception) {
            if (running) Log.e(TAG, "audio renderer worker failed", error)
        } finally {
            release()
        }
    }

    private fun configureCodec(mime: String) {
        val mediaFormat = MediaFormat().apply {
            setString(MediaFormat.KEY_MIME, mime)
            setInteger(MediaFormat.KEY_SAMPLE_RATE, format.sampleRate)
            setInteger(MediaFormat.KEY_CHANNEL_COUNT, format.channels)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 64 * 1024)
            if (mime == MediaFormat.MIMETYPE_AUDIO_AAC) {
                setInteger(MediaFormat.KEY_IS_ADTS, 1)
                setByteBuffer("csd-0", ByteBuffer.wrap(aacAudioSpecificConfig()))
            } else {
                setByteBuffer("csd-0", ByteBuffer.wrap(opusHead()))
                setByteBuffer("csd-1", ByteBuffer.wrap(opusCodecDelay()))
                setByteBuffer("csd-2", ByteBuffer.wrap(opusSeekPreRoll()))
            }
        }
        if (mime == MediaFormat.MIMETYPE_AUDIO_AAC) {
            Log.i(
                TAG,
                "audio AAC config rate=${format.sampleRate} channels=${format.channels} " +
                    "csd0=${aacAudioSpecificConfig().toHexString()}",
            )
        }
        codec = try {
            MediaCodec.createDecoderByType(mime).also {
                it.configure(mediaFormat, null, null, 0)
                it.start()
                Log.i(TAG, "audio decoder configured mime=$mime name=${it.name}")
            }
        } catch (error: Exception) {
            Log.e(TAG, "audio decoder configuration failed mime=$mime", error)
            null
        }
    }

    private fun createTrack() {
        val encoding = AndroidAudioFormat.ENCODING_PCM_16BIT
        val channelMask = if (format.channels >= 2) AndroidAudioFormat.CHANNEL_OUT_STEREO
        else AndroidAudioFormat.CHANNEL_OUT_MONO
        val minBuffer = AudioTrack.getMinBufferSize(format.sampleRate, channelMask, encoding)
        if (minBuffer <= 0) {
            Log.e(TAG, "AudioTrack buffer size unavailable rate=${format.sampleRate} channels=${format.channels}")
            return
        }
        val bufferBytes = maxOf(minBuffer * 4, MIN_TRACK_BUFFER_BYTES)
        startThresholdBytes = if (format.audioType == "telephony" || format.audioType == "speechrecognition") {
            maxOf(minBuffer, MIN_START_BUFFER_BYTES)
        } else {
            maxOf(minBuffer, MIN_START_BUFFER_BYTES)
        }
        track = AudioTrack.Builder()
            .setAudioAttributes(audioAttributes())
            .setAudioFormat(
                AndroidAudioFormat.Builder()
                    .setEncoding(encoding)
                    .setSampleRate(format.sampleRate)
                    .setChannelMask(channelMask)
                    .build(),
            )
            .setBufferSizeInBytes(bufferBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        Log.i(
            TAG,
            "audio track prepared type=${format.payloadType} audioType=${format.audioType} " +
                "codec=${format.codec} " +
                "rate=${format.sampleRate} channels=${format.channels}",
        )
    }

    private fun aacAudioSpecificConfig(): ByteArray {
        val frequencyIndex = MediaCodecSupport.aacFrequencyIndex(format.sampleRate)
        val value = (AAC_OBJECT_TYPE_LC shl 11) or
            (frequencyIndex shl 7) or
            (format.channels.coerceIn(1, 7) shl 3)
        return byteArrayOf((value ushr 8).toByte(), value.toByte())
    }

    private fun audioAttributes(): AudioAttributes {
        val mode = if (advancedAudioChannelMapping) {
            AudioChannelMappingMode.AUTOMOTIVE_BUS
        } else {
            AudioChannelMappingMode.MOBILE_COMPATIBLE
        }
        val selection = AudioChannelMapper.map(
            audioType = format.audioType,
            payloadType = format.payloadType,
            mode = mode,
        )
        val usage = usageFor(selection.channel)
        val contentType = contentTypeFor(selection.contentType)
        return AudioAttributes.Builder()
            .setUsage(usage)
            .setContentType(contentType)
            .build()
            .also {
                Log.i(
                    TAG,
                    "audio route type=${format.payloadType} audioType=${format.audioType} " +
                        "mode=$mode channel=${selection.channel} " +
                        "usage=$usage contentType=$contentType",
                )
            }
    }

    private fun usageFor(channel: AudioChannel): Int = when (channel) {
        AudioChannel.MEDIA -> AudioAttributes.USAGE_MEDIA
        AudioChannel.PHONE -> AudioAttributes.USAGE_VOICE_COMMUNICATION
        AudioChannel.ASSISTANT -> AudioAttributes.USAGE_ASSISTANT
        AudioChannel.NAVIGATION -> AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE
    }

    private fun contentTypeFor(contentType: AudioContentType): Int = when (contentType) {
        AudioContentType.MUSIC -> AudioAttributes.CONTENT_TYPE_MUSIC
        AudioContentType.SPEECH -> AudioAttributes.CONTENT_TYPE_SPEECH
    }

    /** Minimal OpusHead CSD for the mono 48 kHz stream CarPlay negotiates. */
    private fun opusHead(): ByteArray {
        val head = ByteArray(19)
        "OpusHead".toByteArray(Charsets.US_ASCII).copyInto(head, 0)
        head[8] = 1
        head[9] = format.channels.toByte()
        head[10] = 0x38
        head[11] = 0x01
        head[12] = format.sampleRate.toByte()
        head[13] = (format.sampleRate ushr 8).toByte()
        head[14] = (format.sampleRate ushr 16).toByte()
        head[15] = (format.sampleRate ushr 24).toByte()
        return head
    }

    private fun opusCodecDelay(): ByteArray =
        java.nio.ByteBuffer.allocate(8)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .putLong(OPUS_CODEC_DELAY_NANOS)
            .array()

    private fun opusSeekPreRoll(): ByteArray =
        java.nio.ByteBuffer.allocate(8)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .putLong(OPUS_SEEK_PRE_ROLL_NANOS)
            .array()

    private fun handle(packet: AudioPacket) {
        val rtp = packet.rtp
        val timestampUs = sampleTimestampUs(packet.sample)
        when (format.codec) {
            AudioCodecKind.LPCM -> writePcm(byteSwapS16(rtp.copyOfRange(12, rtp.size)))
            AudioCodecKind.AAC_LC -> {
                val accessUnit = rtp.copyOfRange(12, rtp.size)
                if (accessUnit.isNotEmpty()) {
                    if (!firstAacPayloadLogged) {
                        firstAacPayloadLogged = true
                        Log.i(
                            TAG,
                            "audio AAC access unit bytes=${accessUnit.size} " +
                                "head=${accessUnit.copyOf(minOf(accessUnit.size, 16)).toHexString()}",
                        )
                    }
                    feedCodec(
                        MediaCodecSupport.adtsFrame(accessUnit, format.sampleRate, format.channels),
                        timestampUs,
                    )
                }
            }
            AudioCodecKind.OPUS -> {
                val accessUnit = rtp.copyOfRange(12, rtp.size)
                if (accessUnit.size < MIN_OPUS_PACKET_BYTES) {
                    if (!firstOpusShortPacketLogged) {
                        firstOpusShortPacketLogged = true
                        Log.i(
                            TAG,
                            "audio Opus skipping short packet bytes=${accessUnit.size} " +
                                "head=${accessUnit.toHexString()}",
                        )
                    }
                    return
                }
                feedCodec(accessUnit, timestampUs)
            }
        }
    }

    /**
     * Waits up to [AUDIO_INPUT_WAIT_TIMEOUT_US] for a decoder input buffer so a
     * brief stall drops the packet instead of punching an audible hole in the
     * stream.
     */
    private fun awaitDecoderInput(codec: MediaCodec): Int {
        val deadline = System.nanoTime() + AUDIO_INPUT_WAIT_TIMEOUT_US * 1_000
        while (running) {
            val index = codec.dequeueInputBuffer(INPUT_TIMEOUT_US)
            if (index >= 0) return index
            if (System.nanoTime() >= deadline) return -1
        }
        return -1
    }

    private fun sampleTimestampUs(sample: Int): Long =
        (sample.toLong() and 0xffff_ffffL) * 1_000_000L / format.sampleRate

    private fun feedCodec(payload: ByteArray, presentationTimeUs: Long) {
        val codec = codec ?: return
        val index = awaitDecoderInput(codec)
        if (index < 0) {
            inputDropped++
            if (inputDropped == 1) {
                Log.w(
                    TAG,
                    "audio decoder input unavailable codec=${format.codec} " +
                        "queued=$inputQueued dropped=$inputDropped",
                )
            }
            return
        }
        val input = codec.getInputBuffer(index) ?: return
        input.clear()
        if (payload.size <= input.remaining()) {
            input.put(payload)
            codec.queueInputBuffer(index, 0, payload.size, presentationTimeUs, 0)
            inputQueued++
            if (!firstInputQueuedLogged) {
                firstInputQueuedLogged = true
                Log.i(
                    TAG,
                    "audio decoder first input codec=${format.codec} bytes=${payload.size} " +
                        "head=${payload.copyOf(minOf(payload.size, 16)).toHexString()}",
                )
            }
        } else {
            codec.queueInputBuffer(index, 0, 0, 0, 0)
            inputDropped++
        }
        drainCodec(codec)
    }

    private fun drainCodec(codec: MediaCodec) {
        val info = MediaCodec.BufferInfo()
        while (running) {
            val index = codec.dequeueOutputBuffer(info, 0)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> return
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                index >= 0 -> {
                    val size = info.size
                    if (size > 0) {
                        outputBuffers++
                        if (outputBuffers == 1 || outputBuffers % DECODED_BUFFER_LOG_INTERVAL == 0) {
                            Log.i(
                                TAG,
                                "audio decoder output codec=${format.codec} " +
                                    "buffers=$outputBuffers bytes=$size " +
                                    "queued=$inputQueued dropped=$inputDropped",
                            )
                        }
                    }
                    if (size > 0) {
                        val output = codec.getOutputBuffer(index)
                        if (output != null) {
                            if (size > pcm.size) pcm = ByteArray(size)
                            output.position(info.offset)
                            output.limit(info.offset + size)
                            output.get(pcm, 0, size)
                            writePcm(pcm, 0, size)
                        }
                    }
                    codec.releaseOutputBuffer(index, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
                else -> return
            }
        }
    }

    private fun writePcm(data: ByteArray, offset: Int = 0, length: Int = data.size) {
        val track = track ?: return
        if (!firstPcmLogged && length > 0) {
            firstPcmLogged = true
            val end = minOf(data.size, offset + minOf(length, 16))
            Log.i(
                TAG,
                "audio first PCM type=${format.payloadType} bytes=$length " +
                    "head=${data.copyOfRange(offset, end).toHexString()}",
            )
        }
        if (!fadeApplied) {
            applyFadeIn(data, offset, length)
            fadeApplied = true
        }
        var written = 0
        while (written < length && running) {
            val writeLength = if (playbackStarted) {
                length - written
            } else {
                minOf(length - written, PREBUFFER_WRITE_CHUNK_BYTES)
            }
            val count = track.write(data, offset + written, writeLength, AudioTrack.WRITE_BLOCKING)
            if (count <= 0) break
            written += count
            if (!playbackStarted) {
                prebufferBytes += count
                if (prebufferBytes >= startThresholdBytes) {
                    track.play()
                    playbackStarted = true
                    Log.i(TAG, "audio playback started type=${format.payloadType}")
                }
            }
        }
    }

    private fun applyFadeIn(data: ByteArray, offset: Int, length: Int) {
        val samples = (length - length % 2) / 2
        val fadeSamples = minOf(samples, maxOf(1, format.sampleRate / 100))
        for (index in 0 until fadeSamples) {
            val position = offset + index * 2
            val sample = (data[position].toInt() and 0xff) or (data[position + 1].toInt() shl 8)
            val scaled = (sample.toLong() * (index + 1) / fadeSamples).toInt()
            data[position] = scaled.toByte()
            data[position + 1] = (scaled shr 8).toByte()
        }
    }

    private fun byteSwapS16(source: ByteArray): ByteArray {
        for (index in 0 until source.size - 1 step 2) {
            val tmp = source[index]
            source[index] = source[index + 1]
            source[index + 1] = tmp
        }
        return source
    }

    @Synchronized
    private fun release() {
        val codec = codec
        this.codec = null
        if (codec != null) {
            try {
                codec.stop()
            } catch (_: Exception) {
                // Best effort.
            }
            try {
                codec.release()
            } catch (_: Exception) {
                // Best effort.
            }
        }
        val track = track
        this.track = null
        if (track != null) {
            try {
                track.pause()
            } catch (_: Exception) {
                // Best effort.
            }
            try {
                track.flush()
            } catch (_: Exception) {
                // Best effort.
            }
            try {
                track.release()
            } catch (_: Exception) {
                // Best effort.
            }
        }
    }

    private companion object {
        const val TAG = "xcertplay-usb"
        const val AAC_OBJECT_TYPE_LC = 2
        const val MIN_OPUS_PACKET_BYTES = 4
        const val OPUS_CODEC_DELAY_NANOS = 6_500_000L
        const val OPUS_SEEK_PRE_ROLL_NANOS = 80_000_000L
        const val INPUT_TIMEOUT_US = 10_000L
        const val AUDIO_INPUT_WAIT_TIMEOUT_US = 200_000L
        const val MAX_QUEUED_PACKETS = 80
        const val MIN_TRACK_BUFFER_BYTES = 64 * 1024
        const val MIN_START_BUFFER_BYTES = 4 * 1024
        const val PREBUFFER_WRITE_CHUNK_BYTES = 2 * 1024
        const val DECODED_BUFFER_LOG_INTERVAL = 50
    }
}
