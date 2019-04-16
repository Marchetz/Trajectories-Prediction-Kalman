import xml.etree.ElementTree as Et


class XMLParser:

    def __init__(self):
        pass

    @staticmethod
    def parse(path):
        root = Et.parse(path).getroot()[0]  # list of tracklets
        tracklets = {}
        num_of_keys = {}
        frame_interval = {}
        for child in root:
            if child.tag == 'item':  # items are tracklets
                key = child.find('objectType').text  # get the tracklet object type (car, pedestrian, ..)
                if key not in num_of_keys:
                    num_of_keys[key] = 0
                num_of_keys[key] += 1
                key = key + str(num_of_keys[key] - 1)

                poses_list = []
                frame_count = 0
                for pose in child.find('poses'):  # get all the poses. Construct list of tuples.
                    if pose.tag == 'item':
                        pose_tuple = (float(pose.find('tx').text), float(pose.find('ty').text),
                                      float(pose.find('tz').text))
                        poses_list.append(pose_tuple)
                    if pose.tag == 'count':
                        frame_count = int(pose.text)

                tracklets[key] = poses_list
                first_frame = int(child.find('first_frame').text)
                frame_interval[key] = (first_frame, first_frame + frame_count - 1)

        return tracklets, frame_interval
