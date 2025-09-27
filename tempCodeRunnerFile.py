def detect_red_areas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Strict red filter (blood-like)
    lower_deep_red1 = np.array([0, 180, 100])
    upper_deep_red1 = np.array([5, 255, 255])
    lower_deep_red2 = np.array([175, 180, 100])
    upper_deep_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_deep_red1, upper_deep_red1)
    mask2 = cv2.inRange(hsv, lower_deep_red2, upper_deep_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.GaussianBlur(red_mask, (7, 7), 0)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours if contours else []
