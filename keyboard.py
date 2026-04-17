import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller

# Webcam setup
webcam = cv2.VideoCapture(0)
webcam.set(3, 1280)  # Set width
webcam.set(4, 720)   # Set height

# Hand detector setup
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Virtual keyboard setup
keys = [["A", "Z", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["Q", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

finalText = ""
keyboard = Controller()

# Button class for virtual keys
class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

# Create buttons for the keyboard
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# Draw all buttons on the image
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Main loop
while True:
    success, img = webcam.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror the image
    hands, img = detector.findHands(img)  # Detect hands and get landmarks

    img = drawAll(img, buttonList)  # Draw the keyboard

    if hands:
        hand = hands[0]  # Get the first detected hand
        lmList = hand['lmList']  # Get list of landmarks

        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            # Check if index finger is over a button
            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                # Highlight the button
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                # Calculate distance between index and middle fingers
                distance, _, _ = detector.findDistance(lmList[8], lmList[12], img)

                # Simulate key press if fingers are close enough
                if distance < 30:
                    keyboard.press(button.text)
                    keyboard.release(button.text)  # Release the key
                    finalText += button.text
                    sleep(0.2)

    # Display typed text
    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("AI Keyboard", img)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Exit on pressing 'q'
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
