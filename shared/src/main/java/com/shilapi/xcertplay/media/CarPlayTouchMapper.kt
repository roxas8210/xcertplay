package com.shilapi.xcertplay.media

import android.view.MotionEvent
import com.shilapi.xcertplay.airplay.AirPlayContact

/**
 * Converts Android MotionEvents into normalized CarPlay touch contacts.
 *
 * Slots are assigned from stable Android pointer ids so a surviving finger keeps
 * its contact id when another finger lifts; Android pointer *indices* are not
 * stable across lifts and would strand a finger down on the iOS side.
 */
class CarPlayTouchMapper(private val maxContacts: Int = DEFAULT_MAX_CONTACTS) {
    private val slotPointerIds = IntArray(maxContacts) { NO_POINTER }

    fun contacts(event: MotionEvent, viewWidth: Int, viewHeight: Int): List<AirPlayContact> {
        val width = viewWidth.coerceAtLeast(1)
        val height = viewHeight.coerceAtLeast(1)
        val action = event.actionMasked
        if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_CANCEL) {
            // A new gesture starts with DOWN; CANCEL abandons the current one.
            reset()
        }
        val liftedIndex = if (action == MotionEvent.ACTION_POINTER_UP) event.actionIndex else -1
        val allUp = action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL
        val contacts = ArrayList<AirPlayContact>(maxContacts)
        for (index in 0 until event.pointerCount) {
            val pointerId = event.getPointerId(index)
            val lifting = allUp || index == liftedIndex
            val slot = slotFor(pointerId, hold = !lifting)
            if (slot < 0) continue
            contacts.add(
                AirPlayContact(
                    id = slot,
                    x = (event.getX(index).toDouble() / width).coerceIn(0.0, 1.0),
                    y = (event.getY(index).toDouble() / height).coerceIn(0.0, 1.0),
                    down = !lifting,
                ),
            )
        }
        return contacts
    }

    fun reset() {
        slotPointerIds.fill(NO_POINTER)
    }

    /** Returns the slot holding [pointerId], assigning a free one when [hold]; frees it otherwise. */
    private fun slotFor(pointerId: Int, hold: Boolean): Int {
        val existing = slotPointerIds.indexOf(pointerId)
        if (existing >= 0) {
            if (!hold) slotPointerIds[existing] = NO_POINTER
            return existing
        }
        if (!hold) return -1
        val free = slotPointerIds.indexOf(NO_POINTER)
        if (free < 0) return -1
        slotPointerIds[free] = pointerId
        return free
    }

    private companion object {
        const val DEFAULT_MAX_CONTACTS = 2
        const val NO_POINTER = -1
    }
}
