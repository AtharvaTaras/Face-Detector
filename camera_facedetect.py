import cv2

# Global Vars
clr = (0, 0, 0)
thick = 5
face_count = 0

dataset = cv2.CascadeClassifier('frontfacedata.xml')
vid = cv2.VideoCapture(0)

def greyscale():
    global frame, greyed

    ret, frame = vid.read()
    greyed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def get_faces():
    global face_coordinates, greyed

    face_coordinates = dataset.detectMultiScale(greyed)
    print(face_coordinates)


def draw():
    global face_coordinates, clr, thick

    for each in face_coordinates:
        x = each[0]
        y = each[1]
        w = each[2]
        h = each[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), clr, thick)


def display():
    global frame

    cv2.imshow('Camera', frame)
    cv2.waitKey(1)


while True:
    greyscale()
    get_faces()
    draw()
    display()

