import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplex=1, detectionCon=0.7, trackCon=0.7):
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)
        # print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(imgRgb, i, self.mpHands.HAND_CONNECTIONS)
        return imgRgb

    def findPositions(self, img, handNo=0, draw=True):
        handcord = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in list(enumerate(myhand.landmark)):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                handcord.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return handcord


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        if success:
            cv2.imshow('Image', img)
            if cv2.waitKey(20) & 0xff == ord('a'):
                break


main()
