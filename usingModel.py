
from keras.models import load_model
import numpy as np
import cv2
import serial
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Các cài đặt cho kết nối với Arduino
SERIAL_PORT = 'COM3'  # Thay 'X' bằng số cổng COM của Arduino
BAUD_RATE = 9600
def getLetter(result):
  classLabels = {0: 'ok',1: 'up',2: 'call_me',3: 'rock_on',4: 'peace',5: 'fingers_crossed', 6: 'thumbs',7: 'rock', 8:'scissor',9:'paper',
                 }
  try:
    res = int(result)
    return classLabels[res]
  except:
    return "Error"

model = load_model('handgest_200.hdf5')

cap = cv2.VideoCapture(0)
# Kết nối với Arduino
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except:
    print("Failed to connect to Arduino.")
while True:

  ret, frame = cap.read()

  frame = cv2.flip(frame, 1)

  roi = frame[100:400, 320:620]
  roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  ret, thresh2 = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
  thresh2 = np.repeat(np.array(thresh2)[:, :, np.newaxis], 3, axis=2)

  cv2.imshow('roi', thresh2)
  cv2.imwrite('processed_image.jpg', thresh2)

  image = load_img("processed_image.jpg",target_size=(60,60))

  image = img_to_array(image)
  image = image / 255.0
  prediction_image = np.array(image)
  prediction_image = np.expand_dims(image, axis=0)

  prediction = model.predict(prediction_image)
  value = np.argmax(prediction)
  print(value)
  move_name = getLetter(value)
  # print("Prediction is {}.".format(move_name))
  # time.sleep(0.1)

  cv2.putText(frame, move_name, (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
  cv2.imshow('frame', frame)

  # Truyền dự đoán tới Arduino qua Serial
  move_name += '\n'  # Thêm ký tự xuống dòng để Arduino nhận biết kết thúc dữ liệu
  arduino.write(move_name.encode())

  cv2.putText(frame, move_name, (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) == 13:  # 13 is the Enter Key
      break

# Đóng kết nối với Arduino và giải phóng bộ nhớ
arduino.close()
cap.release()
cv2.destroyAllWindows()