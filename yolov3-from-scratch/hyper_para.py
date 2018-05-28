class configure:
    batch_size = 1
    confidence = 0.5
    nms_thresh = 0.4
    num_classes = 80   #coco's 80classes
    with open("data/coco.names", "r") as f:
        classes_name = f.read().split("\n")[:-1]
    img_height = 416

cfg = configure()