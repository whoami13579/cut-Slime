import cv2
import numpy as np
import random
import math
import sys
import mediapipe
from collections import Counter

# ------------------------------------------- 

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


def hand_detection():
    global finger, fingers, a, up, down
    ret, frame = cap.read()
    frame1 = cv2.resize(frame, (WIDTH, HEIGHT))
    a = findpostion(frame1)

    if len(a) != 0:
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
    print(a)


a = None
up = None
down = None
def play():
    global score, life, canvas, a, up, down
    if len(a) != 0:
        x, y = a[9][1], a[9][2]
        x = WIDTH - x

        if up >= 3:
            if WIDTH - 100 < x and x < WIDTH and 0 < y and y < 100:
                cv2.waitKey(0)
            cv2.circle(canvas, (x, y), 10, (255, 0, 0), -1)
        else:
            draw_knife(x, y)
            cut(x, y)


def draw_knife(x, y):
    global knife_img, canvas
    new_center_x, new_center_y = 50, 50

    # 將物件的位置和尺寸轉換為整數
    y1, y2 = max(0, int(y - new_center_y)), min(height, int(y + 50 - new_center_y))
    x1, x2 = max(0, int(x - new_center_x)), min(width, int(x + 50 - new_center_x))

    # 檢查物件位置是否在畫面內
    if y1 < height and y2 > 0 and x1 < width and x2 > 0:
        # 計算圖片和畫布的重疊區域
        img_y1 = max(0, new_center_y - int(y) + y1)
        img_y2 = img_y1 + (y2 - y1)
        img_x1 = max(0, new_center_x - int(x) + x1)
        img_x2 = img_x1 + (x2 - x1)

        # 創建遮罩和反遮罩
        alpha_s = knife_img[img_y1:img_y2, img_x1:img_x2, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            canvas[y1:y2, x1:x2, c] = (alpha_s * knife_img[img_y1:img_y2, img_x1:img_x2, c] +
                                        alpha_l * canvas[y1:y2, x1:x2, c])


def draw_pause(x, y):
    global pause_img, canvas
    new_center_x, new_center_y = 50, 50

    # 將物件的位置和尺寸轉換為整數
    y1, y2 = max(0, int(y - new_center_y)), min(height, int(y + 50 - new_center_y))
    x1, x2 = max(0, int(x - new_center_x)), min(width, int(x + 50 - new_center_x))

    # 檢查物件位置是否在畫面內
    if y1 < height and y2 > 0 and x1 < width and x2 > 0:
        # 計算圖片和畫布的重疊區域
        img_y1 = max(0, new_center_y - int(y) + y1)
        img_y2 = img_y1 + (y2 - y1)
        img_x1 = max(0, new_center_x - int(x) + x1)
        img_x2 = img_x1 + (x2 - x1)

        # 創建遮罩和反遮罩
        alpha_s = pause_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            canvas[:100, WIDTH - 100:, c] = (alpha_s * pause_img[:, :, c] +
                                        alpha_l * canvas[:100, WIDTH - 100:, c])

def cut(x, y):
    global score, life
    for obj in objects:
        h, w, _ = obj.img.shape
        h, w = int(h), int(w)
        # 计算旋转后的图像位置和尺寸
        M = cv2.getRotationMatrix2D((obj.center_x, obj.center_y), obj.angle, 1)
        rotated_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        rotated_corners = cv2.transform(np.array([rotated_corners]), M)[0]
        min_x, min_y = np.min(rotated_corners, axis=0)
        max_x, max_y = np.max(rotated_corners, axis=0)
        y1, y2 = int(obj.y + min_y), int(obj.y + max_y)
        x1, x2 = int(obj.x + min_x), int(obj.x + max_x)

        if y1 <= y <= y2 and x1 <= x <= x2:  # 檢查滑鼠點擊位置是否在物件範圍內
            if obj.value > 0:
                if obj.life > 1:  # 如果物件有多條生命
                    obj.life -= 1  # 減少生命
                    if obj.hit_img is not None:  # 如果有受擊圖片，替換圖片
                        obj.img = obj.hit_img
                        obj.hit_timer = obj.hit_duration  # 设置受击计时器
                else:
                    score += 10  # 水果加分
                    explosions.append(Explosion(obj.x, obj.y))  # 添加爆炸效果
                    objects.remove(obj)  # 從列表中移除被切中的物件
            else:
                life -= 1  # 炸彈扣血
                explosions.append(Explosion(obj.x, obj.y))  # 添加爆炸效果
                objects.remove(obj)
            break
# ------------------------------------------- 

# 初始化遊戲參數
def initialize_game():
    global score, life, speed, gravity, objects, spawn_interval, last_spawn_time, level, explosions
    score = 0
    life = 3
    speed = 1
    gravity = 0.1  # 重力加速度
    objects = []
    spawn_interval = 2  # 初始生成間隔時間（秒）
    last_spawn_time = 0
    level = 1
    explosions = []  # 存储爆炸效果

# 載入背景圖片
canvas = cv2.imread("background.png")
if canvas is None:
    print("Error: Unable to load background image.")
    sys.exit()

height, width, _ = canvas.shape

# 載入水果和炸彈圖片
slime_img = cv2.imread('Slime.png', cv2.IMREAD_UNCHANGED)
if slime_img is None:
    print("Error: Unable to load Slime image.")
    sys.exit()

# 載入紅色史萊姆圖片
red_slime_img = cv2.imread('Red_Slime.png', cv2.IMREAD_UNCHANGED)
if red_slime_img is None:
    print("Error: Unable to load Red Slime image.")
    sys.exit()

# 載入紅色史萊姆受擊圖片
red_slime_hit_img = cv2.imread('Red_Slime_Hit.png', cv2.IMREAD_UNCHANGED)
if red_slime_hit_img is None:
    print("Error: Unable to load Red Slime Hit image.")
    sys.exit()

bomb_img = cv2.imread('bomb.png', cv2.IMREAD_UNCHANGED)
if bomb_img is None:
    print("Error: Unable to load bomb image.")
    sys.exit()

# 載入爆炸圖片
explosion_img = cv2.imread('explosion.png', cv2.IMREAD_UNCHANGED)
if explosion_img is None:
    print("Error: Unable to load explosion image.")
    sys.exit()

knife_img = cv2.imread('knife.png', cv2.IMREAD_UNCHANGED)
if bomb_img is None:
    print("Error: Unable to load knife image.")
    sys.exit()
knife_img = cv2.resize(knife_img, (100, 100))

pause_img = cv2.imread('pause.png', cv2.IMREAD_UNCHANGED)
if bomb_img is None:
    print("Error: Unable to load x image.")
    sys.exit()
pause_img = cv2.resize(pause_img, (100, 100))

# 定義水果和炸彈類別
class Fruit:
    def __init__(self, img, value, hit_img=None, life=1):
        self.img = img
        self.original_img = img  # 保存原始图片
        self.value = value
        self.hit_img = hit_img  # 新增受擊圖片屬性
        self.life = life  # 新增血量屬性
        self.hit_timer = 0  # 受击计时器
        self.hit_duration = 0.5  # 受击状态持续时间（秒）
        self.rotation_angle = random.uniform(-5, 5)  # 随机旋转角度
        self.angle = 0  # 初始旋转角度
        self.spawn_side = random.choice(['bottom', 'left', 'right'])
        img_h, img_w, _ = self.img.shape
        self.center_x, self.center_y = img_w // 2, img_h // 2

        if self.spawn_side == 'bottom':
            self.x = random.randint(self.center_x, width - self.center_x)
            self.y = height + self.center_y
            self.angle_direction = random.uniform(5 * math.pi / 6, math.pi / 6)  # 150 to 30 degrees
        elif self.spawn_side == 'left':
            self.x = -self.center_x
            self.y = random.randint(self.center_y, height - self.center_y)
            self.angle_direction = random.uniform(-math.pi / 4, math.pi / 4)  # -45 to 45 degrees
        elif self.spawn_side == 'right':
            self.x = width + self.center_x
            self.y = random.randint(self.center_y, height - self.center_y)
            self.angle_direction = random.uniform(3 * math.pi / 4, 5 * math.pi / 4)  # 135 to 225 degrees
        self.speed = random.uniform(3, 7) * speed
        self.vx = self.speed * math.cos(self.angle_direction)
        self.vy = -self.speed * math.sin(self.angle_direction)
        self.t = 0  # 時間初始化

    def move(self):
        self.t += 0.1  # 更新時間
        self.x += self.vx
        self.y += self.vy + 0.5 * gravity * self.t ** 2  # 更新y位置，考慮重力影響
        self.angle += self.rotation_angle  # 更新旋转角度

        # 如果在受击状态，更新计时器
        if self.hit_timer > 0:
            self.hit_timer -= 0.1
            if self.hit_timer <= 0:
                self.img = self.original_img  # 恢复原始图片

    def draw(self, canvas):
        h, w, _ = self.img.shape

        # 创建旋转矩阵
        M = cv2.getRotationMatrix2D((self.center_x, self.center_y), self.angle, 1)
        # 计算旋转后的图像的边界框
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - self.center_x
        M[1, 2] += (new_h / 2) - self.center_y
        rotated_img = cv2.warpAffine(self.img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # 更新中心点位置
        new_center_x, new_center_y = new_w // 2, new_h // 2

        # 將物件的位置和尺寸轉換為整數
        y1, y2 = max(0, int(self.y - new_center_y)), min(height, int(self.y + new_h - new_center_y))
        x1, x2 = max(0, int(self.x - new_center_x)), min(width, int(self.x + new_w - new_center_x))

        # 檢查物件位置是否在畫面內
        if y1 < height and y2 > 0 and x1 < width and x2 > 0:
            # 計算圖片和畫布的重疊區域
            img_y1 = max(0, new_center_y - int(self.y) + y1)
            img_y2 = img_y1 + (y2 - y1)
            img_x1 = max(0, new_center_x - int(self.x) + x1)
            img_x2 = img_x1 + (x2 - x1)

            # 創建遮罩和反遮罩
            alpha_s = rotated_img[img_y1:img_y2, img_x1:img_x2, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                canvas[y1:y2, x1:x2, c] = (alpha_s * rotated_img[img_y1:img_y2, img_x1:img_x2, c] +
                                           alpha_l * canvas[y1:y2, x1:x2, c])

# 爆炸效果類別
class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = explosion_img
        self.scale = 1.0
        self.alpha = 1.0
        self.duration = 1.0  # 爆炸效果持续时间（秒）
        self.current_time = 0

    def draw(self, canvas):
        if self.current_time < self.duration:
            h, w, _ = self.img.shape
            scaled_img = cv2.resize(self.img, (int(w * self.scale), int(h * self.scale)))
            scaled_h, scaled_w, _ = scaled_img.shape

            y1, y2 = max(0, int(self.y - scaled_h // 2)), min(height, int(self.y + scaled_h // 2))
            x1, x2 = max(0, int(self.x - scaled_w // 2)), min(width, int(self.x + scaled_w // 2))
            if y1 < height and y2 > 0 and x1 < width and x2 > 0:
                img_y1 = max(0, scaled_h // 2 - int(self.y) + y1)
                img_y2 = img_y1 + (y2 - y1)
                img_x1 = max(0, scaled_w // 2 - int(self.x) + x1)
                img_x2 = img_x1 + (x2 - x1)
                alpha_s = scaled_img[img_y1:img_y2, img_x1:img_x2, 3] / 255.0 * self.alpha
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    canvas[y1:y2, x1:x2, c] = (alpha_s * scaled_img[img_y1:img_y2, img_x1:img_x2, c] +
                                               alpha_l * canvas[y1:y2, x1:x2, c])
            self.scale += 0.1
            self.alpha -= 0.1 / self.duration
            self.current_time += 0.1

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
            # 计算旋转后的图像位置和尺寸
            M = cv2.getRotationMatrix2D((obj.center_x, obj.center_y), obj.angle, 1)
            rotated_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            rotated_corners = cv2.transform(np.array([rotated_corners]), M)[0]
            min_x, min_y = np.min(rotated_corners, axis=0)
            max_x, max_y = np.max(rotated_corners, axis=0)
            y1, y2 = int(obj.y + min_y), int(obj.y + max_y)
            x1, x2 = int(obj.x + min_x), int(obj.x + max_x)

            if y1 <= y <= y2 and x1 <= x <= x2:  # 檢查滑鼠點擊位置是否在物件範圍內
                if obj.value > 0:
                    if obj.life > 1:  # 如果物件有多條生命
                        obj.life -= 1  # 減少生命
                        if obj.hit_img is not None:  # 如果有受擊圖片，替換圖片
                            obj.img = obj.hit_img
                            obj.hit_timer = obj.hit_duration  # 设置受击计时器
                    else:
                        score += 10  # 水果加分
                        explosions.append(Explosion(obj.x, obj.y))  # 添加爆炸效果
                        objects.remove(obj)  # 從列表中移除被切中的物件
                else:
                    life -= 1  # 炸彈扣血
                    explosions.append(Explosion(obj.x, obj.y))  # 添加爆炸效果
                    objects.remove(obj)
                break

# 設定滑鼠事件回呼函數
cv2.namedWindow('Fruit Ninja')
cv2.setMouseCallback('Fruit Ninja', mouse_event)

# 遊戲主迴圈
while True:
    # 清空畫面
    canvas.fill(0)

    hand_detection()
    play()

    # 加載背景圖
    background = cv2.imread("background.png")
    if background is not None:
        canvas = background.copy()

    # 遊戲運行時
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    while life > 0:  # 生命大於0時遊戲繼續
        background = cv2.imread("background.png")
        if background is not None:
            canvas = background.copy()
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        draw_pause(100, 100)
        hand_detection()
        play()

        # 根據分數調整難度
        if score >= level * 50:  # 每達到50分，增加一級難度
            level += 1
            speed *= 1.2  # 增加速度
            spawn_interval *= 0.8  # 減少生成間隔

        # 生成新的物件
        if current_time - last_spawn_time > spawn_interval:
            obj_type = random.choice([
                Fruit(slime_img, 1), 
                Fruit(bomb_img, -1), 
                Fruit(red_slime_img, 1, hit_img=red_slime_hit_img, life=3)  # 添加紅史萊姆
            ])
            objects.append(obj_type)
            last_spawn_time = current_time

        # 移動和繪製現有物件
        for obj in objects:
            obj.move()
            if obj.y - obj.center_y > height or obj.x - obj.center_x > width or obj.y + obj.center_y < 0 or obj.x + obj.center_x < 0:
                if obj.t > 1:  # 確保物件在畫面內存在一段時間後才判斷是否出界
                    if obj.value > 0:  # 只有水果超出畫面才扣生命值
                        life -= 1
                    objects.remove(obj)
            else:
                obj.draw(canvas)

        # 绘制爆炸效果
        for explosion in explosions:
            explosion.draw(canvas)

        # 移除已经结束的爆炸效果
        explosions = [exp for exp in explosions if exp.current_time < exp.duration]

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
    while life <= 0:
        canvas = cv2.imread("background_reel.png")  # 重新加載背景圖
        cv2.putText(canvas, "GAME OVER", (width // 2 - 180, height // 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        cv2.putText(canvas, f"Final Score: {score}", (width // 2 - 125, height // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(canvas, "Press 'W' to continue", (width // 2 - 165, height // 2 + 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        hand_detection()
        play()
        # print("line: 467")
        cv2.imshow('Fruit Ninja', canvas)

        key = cv2.waitKey(30)  # 等待按鍵事件
        if key == ord('w'):
            history_scores.append(score)
            history_scores = sorted(history_scores, reverse=True)[:5]  # 保留最高的五個分數
    
            canvas = cv2.imread("background_reel.png")  # 重新加載背景圖
            cv2.putText(canvas, "History Score", (width // 2 - 205, height // 2 - 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            for i, hs in enumerate(history_scores):
                cv2.putText(canvas, f"{i+1}. {hs}", (width // 2 - 50, height // 2 - 20 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, "Press 'W' to Restart or 'Q' to Quit", (width // 2 - 275, height // 2 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            
            cv2.imshow('Fruit Ninja', canvas)
            
            while True:
                key = cv2.waitKey(30)  # 等待按鍵事件，讓玩家查看歷史分數
                if key == ord('w'):  # 按下W鍵重新開始遊戲
                    initialize_game()
                    break
                if key == ord('q'):  # 按下Q鍵退出遊戲
                    cv2.destroyAllWindows()
                    sys.exit()

    if key == ord('q'):  # 按下Q鍵退出遊戲
        break

cv2.destroyAllWindows()