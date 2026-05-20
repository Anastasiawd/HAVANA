"""
ros_obs_loader.py
-----------------
Subscribe to a ROS Image topic and maintain a rolling context window of the
most recent observation frames.  This module does NOT call ``rospy.init_node``
— the caller (e.g. test.py) is responsible for initialising the ROS node.

Typical usage
~~~~~~~~~~~~~
::

    import rospy
    rospy.init_node("nav_inference")

    loader = ROSObsLoader(
        image_topic="/usb_cam/image_raw",
        context_size=3,
        timeout=5.0,
    )

    # block until context is filled
    obs_images = loader.wait_for_context()
    # obs_images: list[PIL.Image.Image], len == context_size + 1
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np
from PIL import Image as PILImage


class ROSObsLoader:
    """Rolling-window ROS image subscriber.

    Parameters
    ----------
    image_topic : str
        ROS Image topic to subscribe to.
    context_size : int
        Number of *past* context frames.  The buffer holds
        ``context_size + 1`` images total (context + current).
    timeout : float
        Seconds to wait in ``wait_for_context`` before raising.
    encoding : str
        Expected encoding of the incoming Image messages.
        Supports ``"rgb8"``, ``"bgr8"``, ``"mono8"``.
    use_cv_bridge : bool
        If True, use ``cv_bridge`` for decoding.  If False (default),
        decode manually from ``msg.data`` (avoids cv_bridge dependency).
    """

    def __init__(
        self,
        image_topic: str = "/usb_cam/image_raw",
        context_size: int = 3,
        timeout: float = 5.0,
        encoding: str = "rgb8",
        use_cv_bridge: bool = False,
    ):
        import rospy
        from sensor_msgs.msg import Image as ROSImage

        self._topic = image_topic
        self._context_size = context_size
        self._buffer_size = context_size + 1
        self._timeout = timeout
        self._encoding = encoding
        self._use_cv_bridge = use_cv_bridge

        self._lock = threading.Lock()
        self._buffer: List[PILImage.Image] = []

        # optional cv_bridge
        self._bridge = None
        if use_cv_bridge:
            try:
                from cv_bridge import CvBridge
                self._bridge = CvBridge()
            except ImportError:
                raise ImportError(
                    "cv_bridge not found. Set use_cv_bridge=False or install cv_bridge."
                )

        # subscribe
        self._sub = rospy.Subscriber(
            image_topic, ROSImage, self._callback, queue_size=1
        )
        rospy.loginfo(
            f"[ROSObsLoader] Subscribed to '{image_topic}' "
            f"(context_size={context_size})"
        )

    # ------------------------------------------------------------------
    # ROS callback
    # ------------------------------------------------------------------

    def _callback(self, msg) -> None:
        """Convert incoming ROS Image to PIL and append to rolling buffer."""
        pil_img = self._msg_to_pil(msg)
        with self._lock:
            if len(self._buffer) >= self._buffer_size:
                self._buffer.pop(0)
            self._buffer.append(pil_img)

    def _msg_to_pil(self, msg) -> PILImage.Image:
        """Decode a sensor_msgs/Image to a PIL RGB image."""
        if self._bridge is not None:
            import cv2
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return PILImage.fromarray(cv_img)

        # manual decoding (no cv_bridge dependency)
        encoding = msg.encoding.lower() if msg.encoding else self._encoding.lower()
        h, w = msg.height, msg.width

        if encoding in ("rgb8",):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        elif encoding in ("bgr8",):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            arr = arr[:, :, ::-1].copy()  # BGR → RGB
        elif encoding in ("mono8",):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            arr = np.stack([arr] * 3, axis=-1)  # grayscale → RGB
        elif "compressed" in encoding:
            import io
            arr = np.array(PILImage.open(io.BytesIO(msg.data)).convert("RGB"))
        else:
            # fallback: try to interpret as 3-channel
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
            if arr.shape[2] == 4:  # RGBA / BGRA
                arr = arr[:, :, :3]

        return PILImage.fromarray(arr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def context_ready(self) -> bool:
        """Return True when the buffer is fully filled."""
        with self._lock:
            return len(self._buffer) >= self._buffer_size

    def get_obs_context(self) -> Optional[List[PILImage.Image]]:
        """Return current buffer contents, or None if not yet full."""
        with self._lock:
            if len(self._buffer) < self._buffer_size:
                return None
            return list(self._buffer)  # shallow copy

    def wait_for_context(self, poll_hz: float = 10.0) -> List[PILImage.Image]:
        """Block until the context window is full, then return it.

        Parameters
        ----------
        poll_hz : float
            Polling rate (Hz) while waiting.

        Returns
        -------
        list[PIL.Image.Image]
            ``[oldest_context, ..., current_frame]``  (length = context_size + 1)

        Raises
        ------
        TimeoutError
            If the context is not filled within ``self._timeout`` seconds.
        """
        import rospy

        start = time.time()
        rate = rospy.Rate(poll_hz)

        while not rospy.is_shutdown():
            ctx = self.get_obs_context()
            if ctx is not None:
                return ctx
            elapsed = time.time() - start
            if elapsed > self._timeout:
                raise TimeoutError(
                    f"[ROSObsLoader] Timed out after {self._timeout:.1f}s "
                    f"waiting for {self._buffer_size} frames on '{self._topic}' "
                    f"(received {len(self._buffer)})"
                )
            rate.sleep()

        raise RuntimeError("[ROSObsLoader] rospy shutdown before context filled")

    def clear(self) -> None:
        """Flush the buffer so the next ``wait_for_context`` starts fresh."""
        with self._lock:
            self._buffer.clear()

    def shutdown(self) -> None:
        """Unsubscribe from the topic."""
        self._sub.unregister()
