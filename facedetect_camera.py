import cv2

# Global Vars
clr = (0, 0, 0)
thick = 5
font = cv2.FONT_HERSHEY_PLAIN

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


def count():
    global img, font, clr, frame

    face_count = 0
    for each in face_coordinates:
        face_count += 1
        faces = str(face_count)
        text = ('Faces found - ' + faces)
        # img = ...(image, text, coordinates, font, size, color, thickness, ...)
        frame = cv2.putText(frame, text, (25, 45), font, 2, clr, 2, cv2.LINE_AA)


def display():
    global frame

    cv2.imshow('Camera', frame)
    cv2.waitKey(1)


while True:
    greyscale()
    get_faces()
    draw()
    count()
    display()
