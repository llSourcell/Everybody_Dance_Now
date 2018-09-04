import argparse
import PyOpenPose as OP
import cv2
import os
import numpy as np
import tensorflow as tf

CROP_SIZE = 256

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

def resize(image):
    """Crop and resize image for pix2pix."""
    height = image.shape[0]
    width = image.shape[1]
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def main():
    # OpenPose
    with_face = with_hands = True
    op = OP.OpenPose((320, 240), (240, 240), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                      False, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    # TensorFlow
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # OpenCV
    cap = cv2.VideoCapture(args.video_source)

    while True:
        try:
            ret, frame = cap.read()
        except Exception as e:
            print("Failed to grab", e)
            break

        if frame is None:
            continue
        rgb_resize = cv2.resize(frame, (640, 480))

        op.detectPose(rgb_resize)
        op.detectFace(rgb_resize)
        op.detectHands(rgb_resize)

        res = op.render(rgb_resize)
        persons = op.getKeypoints(op.KeypointType.POSE)[0]

        if persons is not None and len(persons) > 1:
            print("First Person: ", persons[0].shape)
            continue

        gray = cv2.cvtColor(res-rgb_resize, cv2.COLOR_RGB2GRAY)
        ret, resize_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        resize_binary = cv2.cvtColor(resize_gray, cv2.COLOR_GRAY2RGB)

        # generate prediction
        combined_image = np.concatenate([resize(resize_binary), resize(rgb_resize)], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        image_normal = np.concatenate([resize(rgb_resize), image_bgr], axis=1)
        image_pose = np.concatenate([resize(resize_binary), image_bgr], axis=1)
        image_all = np.concatenate([resize(rgb_resize), resize(resize_binary), image_bgr], axis=1)

        if args.display == 0:
            cv2.imshow('pose2pose', image_normal)
        elif args.display == 1:
            cv2.imshow("pose2pose", image_pose)
        else:
            cv2.imshow('pose2pose', image_all)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sess.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('--show', dest='display', type=int, default=2, choices=[0, 1, 2],
                        help='0 shows the normal input; 1 shows the pose; 2 shows the normal input and pose')
    parser.add_argument('--tf-model', dest='frozen_model_file', type=str, default='pose2pose-reduced-model/frozen_model.pb',help='Frozen TensorFlow model file.')
    args = parser.parse_args()
    main()
