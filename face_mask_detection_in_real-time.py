import numpy as np
import cv2

L_eye_cascade = cv2.CascadeClassifier('eye_haarcascade.xml')
R_eye_cascade = cv2.CascadeClassifier('eye_haarcascade.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gamma = 1.1
    img = np.clip(np.power(img, gamma), 0, 255).astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    L_eye = L_eye_cascade.detectMultiScale(gray, 1.3, 5)
    print('Left :', L_eye)
    for (x, y, w, h) in L_eye:
        R_eye_check = gray[y - h//2:y + 3*h//2, x + w:x + 3*w]
        R_eye = R_eye_cascade.detectMultiScale(R_eye_check, 1.3, 5)
        print('Right :', R_eye)
        for (a, b, c, d) in R_eye:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(img, (x + w + a, y - h//2 + b), (x + w + a + c, y - h//2 + b + d), (255, 0, 0), 2)
            U, L, D, R = np.max([y+h, y-h//2+b+d]) + a//2, \
                         x+w//2, \
                         np.max([y+h, y-h//2+b+d]) + 3*np.max([h, d])//2, \
                         x+w+a+c//2
            Nose_check = np.float32(gray[U:D, L:R])
            pixel_values = Nose_check.reshape((-1))
            # K-means Classification
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
            _, labels, (centers) = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            labels = labels.flatten()
            segmented_image = centers[labels.flatten()]
            Nose_check2 = segmented_image.reshape(Nose_check.shape)
            # Threshold
            ret, Nose = cv2.threshold(Nose_check2, 127, 255, cv2.THRESH_BINARY)
            # View Nose Region
            cv2.imshow('Nose', Nose)
            ACC = (np.sum(Nose)/255)/(len(Nose) * len(Nose[0]))
            U2, L2, D2, R2 = (y + y - h//2 + b) // 2 - np.max([h, d]), \
                             x - w//2, \
                             (y + y - h//2 + b) // 2 + 3 * np.max([h, d]), \
                             x + w + a + c//2 + c
            print(ACC)
            if ACC >= 0.99:
                cv2.rectangle(img, (L2, U2), (R2, D2), (0, 255, 0), 2)
                print('MASK IS DETECTED!!')
            else:
                cv2.rectangle(img, (L2, U2), (R2, D2), (0, 0, 255), 2)
                print('MASK IS NOT DETECTED!!')
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
