import cv2

# Importing pre-trained face data
dataset = cv2.CascadeClassifier('frontfacedata.xml')

# Picking image to test
img = cv2.imread('testface.jpg')

# Convert to greyscale
greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detection
face_coordinates = dataset.detectMultiScale(greyed)
print(face_coordinates)

# Border Properties
clr = (0, 0, 0)
thick = 5

# Face Counter
face_count = 0

# Draw Border
for each in face_coordinates:
    x = each[0]
    y = each[1]
    w = each[2]
    h = each[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), clr, thick)
    face_count += 1

# Display Face Count
faces = str(face_count)
text = ('Faces found - ' + faces)
font = cv2.FONT_HERSHEY_PLAIN
# img = ...(image, text, coordinates, font, size, color, thickness, ...)
img = cv2.putText(img, text, (25, 45), font, 2, clr, 2, cv2.LINE_AA)

# Output
cv2.imshow('Test', img)
cv2.waitKey()
