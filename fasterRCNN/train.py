import fire

from torch.autograd import Variable
from torch.utils import data as data_
from tqdm import tqdm 

from data.dataset import Dataset, TestDataset
from utils.config import opt
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from utils import array_tool as at 
from utils.eval_tool import eval_detection_voc
from trainer import FasterRCNNTrainer

def train(**kwargs):
	opt._parse(kwargs)
	dataset = Dataset(opt)
	print('load data')

	dataloader = data_.DataLoader(dataset, 
								  batch_size=1,
								  shuffle=True,
								  num_workers=opt.num_workers)
	testset = TestDataset(opt)
	test_dataloader = data_.DataLoader(testset,
									   batch_size=1,
									   num_workers=opt.num_workers,
									   shuffle=False
									   )

	faster_rcnn = FasterRCNNVGG16()
	print('model construct completed')

	trainer = FasterRCNNTrainer(faster_rcnn).cuda()

	if opt.load_path:
		trainer.load(opt.load_path)  # !TODO
		print('load pre-trained model')  

	best_map = 0
	lr_ = opt.lr


	for epoch in range(opt.epoch):
		trainer.reset_meters()
		for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
			scale = at.scalar(scale)
			img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
			img, bbox, label = Variable(img), Variable(bbox), Variable(label)

			trainer.train_step(img, bbox, label, scale)

		eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
		if eval_result['map'] > best_map:
			best_map = eval_result['map']
			best_path = trainer.save(best_map=best_map)

		if epoch == 9:
			trainer.load(best_path)
			trainer.faster_rcnn.scale_lr(opt.lr_decay)
			lr_ = lr_*opt.lr_decay

		if epoch == 13:
			break



def eval(dataloader, faster_rcnn, test_num=10000):
	pred_bboxes, pred_labels, pred_scores = list(), list(), list()
	gt_bboxes, gt_labels = list(), list()

	for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
		sizes = [sizes[0][0], sizes[1][0]]
		pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
		gt_bboxes += list(gt_bboxes_.numpy())
		gt_labels += list(gt_labels_.numpy())

		pred_bboxes += pred_bboxes_
		pred_labels += pred_labels_
		pred_scores += pred_scores_
		if ii == test_num: break

	result = eval_detection_voc(
				pred_bboxes,
				pred_labels,
				pred_scores,
				gt_bboxes,
				gt_labels,
				use_07_metric=True)

	return result 

if __name__ == '__main__':
    fire.Fire()

