import cv2
import numpy as np
import face_recognition as fr

imgElon = fr.load_image_file('elon_musk_train.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest = fr.load_image_file('bill_gates.jpg')
# imgTest = fr.load_image_file('test_elon_musk.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceloc = fr.face_locations(imgElon)[0]
encodeElon = fr.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
# print(faceloc)

facelocTest = fr.face_locations(imgTest)[0]
encodeElonTest = fr.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results=fr.compare_faces([encodeElon],encodeElonTest)
faceDis = fr.face_distance([encodeElon], encodeElonTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Musk',imgTest)
cv2.waitKey(0)