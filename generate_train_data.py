"""
Example script using PyOpenPose.
"""
import PyOpenPose as OP
import time
import cv2
import os

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

def run():

    cap = cv2.VideoCapture(args.filename)
    with_face = with_hands = True
    op = OP.OpenPose((656, 368), (368, 368), (1280, 720), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                      False, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    paused = False
    delay = {True: 0, False: 1}

    count = 0
    print("Entering main Loop.")
    while True:
        try:
            ret, frame = cap.read()
            rgb = frame
        except Exception as e:
            print("Failed to grab", e)
            break

        t = time.time()
        op.detectPose(rgb)
        op.detectFace(rgb)
        op.detectHands(rgb)
        t = time.time() - t

        res = op.render(rgb)
        persons = op.getKeypoints(op.KeypointType.POSE)[0]

        if persons is None:
            print("No Person")
            continue

        if persons is not None and len(persons) > 1:
            print("Person > 1 ", persons[0].shape)
            continue

        gray = cv2.cvtColor(res-rgb, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow("OpenPose result", binary)
        count += 1
        print(count)
        #if count < 155 or (count > 855 and count < 1255) or (count > 1745 and count < 1775) or (count > 2225 and count < 2265) or (count > 2535 and count < 2915):
        #    continue
        #if count > 2405:
        #    break
        cv2.imwrite("original/{}.png".format(count), rgb)
        cv2.imwrite("landmarks/{}.png".format(count), binary)

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    args = parser.parse_args()
    if not os.path.exists(os.path.join('./', 'original')):
        os.makedirs(os.path.join('./', 'original'))
    if not os.path.exists(os.path.join('./', 'landmarks')):
        os.makedirs(os.path.join('./', 'landmarks'))
    # os.makedirs('original', exist_ok=True)
    # os.makedirs('landmarks', exist_ok=True)
    run()
