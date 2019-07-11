import face_recognition

monali_image = face_recognition.load_image_file("data/monali.jpg")
shikha_image = face_recognition.load_image_file("data/shikha.jpg")
unknown_image = face_recognition.load_image_file("data/shikha2.jpg")

try:
    monali_face_encoding = face_recognition.face_encodings(monali_image)[0]
    shikha_face_encoding = face_recognition.face_encodings(shikha_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    monali_face_encoding,
    shikha_face_encoding
]

results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Monali ? {}".format(results[0]))
print("Is the unknown face a picture of Shikha ? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
