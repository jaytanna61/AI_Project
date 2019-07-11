import face_recognition
import cv2

input_movie = cv2.VideoCapture("data/bhargavi_video.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('data/bhargavi_output.avi', fourcc, 30, (720, 1080))

bhargavi_image = face_recognition.load_image_file("data/BHARGAVI.png")
bhargavi_face_encoding = face_recognition.face_encodings(bhargavi_image)[0]

shikha_image = face_recognition.load_image_file("data/SHIKHA.png")
shikha_face_encoding = face_recognition.face_encodings(shikha_image)[0]

known_faces = [
    bhargavi_face_encoding,
    shikha_face_encoding
]

face_names = [
    "BHARGAVI",
    "SHIKHA"
]
face_locations = []
face_encodings = []

frame_number = 0

while True:
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "BHARGAVI"
        elif match[1]:
             name = "SHIKHA"

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()
