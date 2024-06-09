import cv2
import numpy as np
import random
import math
import sys
import mediapipe
from collections import Counter

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

mod = handsModule.Hands(max_num_hands=1)

WIDTH = 1200
HEIGHT = 678

cap = cv2.VideoCapture(0)
tip = [8, 12, 16, 20]
tipname = [8, 12, 16, 20]
fingers = []
finger = []

def findpostion(frame1):
    list = []
    results = mod.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks != None:
        for handLandmarks in results.multi_hand_landmarks:
            drawingModule.draw_landmarks(
                frame1, handLandmarks, handsModule.HAND_CONNECTIONS
            )
            list = []
            for id, pt in enumerate(handLandmarks.landmark):
                x = int(pt.x * WIDTH)
                y = int(pt.y * HEIGHT)
                list.append([id, x, y])

    return list


def findnameoflandmark(frame1):
    list = []
    results = mod.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks != None:
        for handLandmarks in results.multi_hand_landmarks:

            for point in handsModule.HandLandmark:
                list.append(
                    str(point)
                    .replace("< ", "")
                    .replace("HandLandmark.", "")
                    .replace("_", " ")
                    .replace("[]", "")
                )
    return list

# 初始化遊戲參數
def initialize_game():
    global score, life, speed, gravity, objects
    score = 0
    life = 3
    speed = 1
    gravity = 0.1  # 重力加速度
    objects = []

# 載入背景圖片
canvas = cv2.imread("background.png")
if canvas is None:
    print("Error: Unable to load background image.")
    sys.exit()

height, width, _ = canvas.shape

# 定義水果和炸彈類別
class Fruit:
    def __init__(self, img, value):
        self.img = img
        self.value = value
        self.spawn_side = random.choice(['bottom', 'left', 'right'])
        if self.spawn_side == 'bottom':
            self.x = random.randint(50, width - 50)
            self.y = height
            self.angle = random.uniform(-5 * math.pi / 6, -math.pi / 6)  # -150 to -30 degrees
        elif self.spawn_side == 'left':
            self.x = 0
            self.y = random.randint(50, height - 50)
            self.angle = random.uniform(-math.pi / 4, math.pi / 4)  # -45 to 45 degrees
        elif self.spawn_side == 'right':
            self.x = width
            self.y = random.randint(50, height - 50)
            self.angle = random.uniform(3 * math.pi / 4, 5 * math.pi / 4)  # 135 to 225 degrees
        self.speed = random.uniform(3, 7) * speed
        self.vx = self.speed * math.cos(self.angle)
        self.vy = -self.speed * math.sin(self.angle)
        self.t = 0  # 時間初始化

    def move(self):
        self.t += 0.1  # 更新時間
        self.x += self.vx
        self.y += self.vy + 0.5 * gravity * self.t ** 2  # 更新y位置，考慮重力影響

    def draw(self):
        h, w, _ = self.img.shape
        h, w = int(h), int(w)  # 將高度和寬度轉換為整數

        # 將物件的位置和尺寸轉換為整數
        y1, y2 = max(0, int(self.y)), min(height, int(self.y + h))
        x1, x2 = max(0, int(self.x)), min(width, int(self.x + w))

        # 檢查物件位置是否在畫面內
        if y1 < height and y2 > 0 and x1 < width and x2 > 0:
            # 計算在畫布和圖像中相應的切片範圍
            canvas[y1:y2, x1:x2] = self.img[(y1 - int(self.y)):(y2 - int(self.y)), (x1 - int(self.x)):(x2 - int(self.x))]

# 載入水果和炸彈圖片
Slime_img = cv2.imread('Slime.png')
if Slime_img is None:
    print("Error: Unable to load Slime image.")
    sys.exit()

bomb_img = cv2.imread('bomb.png')
if bomb_img is None:
    print("Error: Unable to load bomb image.")
    sys.exit()

knife_img = cv2.imread('knife.png')
if bomb_img is None:
    print("Error: Unable to load knife image.")
    sys.exit()

knife_img = cv2.resize(knife_img, (100, 100))

x_img = cv2.imread('x.png')
if bomb_img is None:
    print("Error: Unable to load x image.")
    sys.exit()

x_img = cv2.resize(x_img, (100, 100))

# 初始化遊戲物件列表
initialize_game()

# 歷史分數列表
history_scores = []

# 滑鼠事件處理函數
def mouse_event(event, x, y, flags, param):
    global score, life
    if event == cv2.EVENT_LBUTTONDOWN:  # 滑鼠左鍵點擊事件
        for obj in objects:
            h, w, _ = obj.img.shape
            h, w = int(h), int(w)
            y1, y2 = int(obj.y), int(obj.y + h)
            x1, x2 = int(obj.x), int(obj.x + w)
            if y1 <= y <= y2 and x1 <= x <= x2:  # 檢查滑鼠點擊位置是否在物件範圍內
                if obj.value > 0:
                    score += 10  # 水果加分
                else:
                    life -= 1  # 炸彈扣血
                objects.remove(obj)  # 從列表中移除被切中的物件
                break

# 設定滑鼠事件回呼函數
cv2.namedWindow('Fruit Ninja')
cv2.setMouseCallback('Fruit Ninja', mouse_event)

# 遊戲主迴圈
while True:
    canvas.fill(0)  # 清空畫面

    # 遊戲運行時
    while life > 0:  # 生命大於0時遊戲繼續
        canvas = cv2.imread("background.png")  # 重新加載背景圖

        canvas[:100, WIDTH - 100:] = x_img[:, :]
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (WIDTH, HEIGHT))
        a = findpostion(frame1)
        b = findnameoflandmark(frame1)

        if len(b and a) != 0:
            finger = []
            if a[0][1:] < a[4][1:]:
                finger.append(1)

            else:
                finger.append(0)

            fingers = []
            for id in range(0, 4):
                if a[tip[id]][2:] < a[tip[id] - 2][2:]:

                    fingers.append(1)

                else:
                    fingers.append(0)
        x = fingers + finger
        c = Counter(x)
        up = c[1]
        down = c[0]

        if len(b and a) != 0:
            x, y = a[9][1], a[9][2]
            x = WIDTH - x

            if up >= 3:
                if WIDTH - 100 < x and x < WIDTH and 0 < y and y < 100:
                    life = 0
                cv2.circle(canvas, (x, y), 10, (255, 0, 0), -1)
            else:
                # cv2.circle(canvas, (x, y), 10, (0, 0, 255), -1)
                y1, y2 = max(0, int(y)), min(height, int(y + 100))
                x1, x2 = max(0, int(x)), min(width, int(x + 100))
                canvas[y1:y2, x1:x2] = knife_img[(y1 - int(y)):(y2 - int(y)), (x1 - int(x)):(x2 - int(x))]

                for obj in objects:
                    h, w, _ = obj.img.shape
                    h, w = int(h), int(w)
                    y1, y2 = int(obj.y), int(obj.y + h)
                    x1, x2 = int(obj.x), int(obj.x + w)
                    if y1 <= y+50 <= y2 and x1 <= x+50 <= x2:  # 檢查滑鼠點擊位置是否在物件範圍內
                        if obj.value > 0:
                            score += 10  # 水果加分
                        else:
                            life -= 1  # 炸彈扣血
                        objects.remove(obj)  # 從列表中移除被切中的物件
                        break


        # 生成新的物件（只產生一個）
        if len(objects) == 0:
            # 隨機選擇要生成的物件類型
            obj_type = random.choice([Fruit(Slime_img, 1), Fruit(bomb_img, -1)])
            objects.append(obj_type)

        # 移動和繪製現有物件
        for obj in objects:
            obj.move()
            if obj.y > height or obj.x > width or obj.y < 0 or obj.x < 0:
                if obj.value > 0:  # 只有水果超出畫面才扣生命值
                    life -= 1
                objects.remove(obj)
            else:
                obj.draw()

        # 顯示遊戲資訊和畫面
        cv2.putText(canvas, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Life: {life}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if history_scores:
            cv2.putText(canvas, f"History Score: {history_scores[0]}", (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Fruit Ninja', canvas)

        key = cv2.waitKey(30)
        if key == 27:  # 按下ESC暫停遊戲
            cv2.waitKey(-1)
        if key == ord('q'):  # 按下Q鍵退出遊戲
            break

    # 游戏结束，显示 GAME OVER
    if life <= 0:
        history_scores.append(score)
        history_scores = sorted(history_scores, reverse=True)[:5]  # 保留最高的五個分數

        canvas = cv2.imread("background.png")  # 重新加載背景圖
        cv2.putText(canvas, "GAME OVER", (width // 2 - 150, height // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        cv2.putText(canvas, f"Final Score: {score}", (width // 2 - 150, height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for i, hs in enumerate(history_scores):
            cv2.putText(canvas, f"{i+1}. {hs}", (width // 2 - 150, height // 2 + 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(canvas, "Press 'W' to Restart or 'Q' to Quit", (width // 2 - 300, height // 2 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Fruit Ninja', canvas)
        
        while True:
            key = cv2.waitKey(0)  # 等待按鍵事件
            if key == ord('w'):  # 按下W鍵重新開始遊戲
                initialize_game()
                break
            if key == ord('q'):  # 按下Q鍵退出遊戲
                cv2.destroyAllWindows()
                sys.exit()

    if key == ord('q'):  # 按下Q鍵退出遊戲
        break

cv2.destroyAllWindows()