import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets, models

from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import time
import matplotlib.pyplot as plt
import argparse


def main(data_dir,out,gpu):
	# Load the data
	transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
	train_data = datasets.ImageFolder(data_dir + '/train', transform = transform)
	test_data = datasets.ImageFolder(data_dir + '/test', transform=  transform)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=500, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False)

	
	model = models.resnet18(pretrained=True)
	num_fc_in = model.fc.in_features
	model.fc = nn.Linear(num_fc_in, 2)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	gpu = 'cuda:'+ str(gpu)
	device = torch.device(gpu if torch.cuda.is_available() else "cpu")
	model.to(device)

	num_epochs = 100
	#Define the lists to store the results of loss and accuracy
	train_acc_epoch = []
	test_acc_epoch = []
	test_acc_0_epoch = []
	test_acc_1_epoch = []

	for epoch in range(num_epochs):  # loop over the dataset multiple times
		print('*' * 10)
		print(f'epoch {epoch + 1}')
		correct_train = 0
		iter_train = 0
		iter_loss = 0.0
		#training mode
		model.train()
		start = time.time()
		for i, data in enumerate(train_loader):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data[0].to(device), data[1].to(device)
			#  print(labels)

	        # zero the parameter gradients
			optimizer.zero_grad()

			# forward + optimize
			outputs = model(inputs)
			# print(outputs)
			loss = criterion(outputs, labels)
			# print(loss)
			iter_loss += float(loss)
			# backward
			loss.backward()
			optimizer.step()

			_, predicted = torch.max(outputs, 1)
			correct_train += (predicted == labels).sum()
			iter_train += 1

		# # Record the training loss
	    # train_loss_epoch.append(iter_loss / iter_train)
	    # Record the training accuracy
		train_acc_epoch.append((correct_train / len(train_data)))


		# Testing
	    # evaluation mode
		model.eval()
		correct_test = 0
		y_true = []
		y_pred = []

		for i, data in enumerate(test_loader):
			inputs, labels = data[0].to(device), data[1].to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)  # Calculate the loss
			# Record the correct predictions for training data
			_, predicted = torch.max(outputs, 1)
			correct_test += (predicted == labels).sum()
			y_true.append(labels)
			y_pred.append(predicted)
	        
		# Record the Testing accuracy
		test_acc_epoch.append(( correct_test / len(test_data)))
		stop = time.time()
		print(
	        'Epoch {}/{}, Training Accuracy: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
	        .format(epoch + 1, num_epochs, train_acc_epoch[-1], test_acc_epoch[-1], stop-start))


		# GPU tensor to cpu tensor
		y_true = torch.cat(y_true,0)
		y_true = y_true.cpu()
		y_pred_class = torch.cat(y_pred,0)
		y_pred_class = y_pred_class.cpu()

		# Test confusion matrix
		cm_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred_class)
		each_class_acc = cm_matrix.diagonal() / cm_matrix.sum(axis=1)
		each_class_ratio = cm_matrix.sum(axis=1) / cm_matrix.sum()
		acc_test_epoch = cm_matrix.diagonal().sum() / cm_matrix.sum()
		print("Overall accuracy is: {:.3f}".format(acc_test_epoch))
		print(cm_matrix)
		print("Each class accuracy:")
		print(each_class_acc)
		print(y_true)
		print(y_pred_class)


	print('Finished Training')
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(train_acc_epoch, label="train acc")
	plt.plot(test_acc_epoch, label="test acc")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend(loc="upper right")
	plt.savefig("./" + out+ ".jpg")

	# Test the network on the test data
	modelname = './' + out + '.pth'
	torch.save(model.state_dict(), modelname)


parser = argparse.ArgumentParser(description="Define the data folder, gpu and out name")
parser.add_argument('--gpu','-g', help='number for gpu:1, 2, 3')
parser.add_argument('--data','-d', help='the data folder')
parser.add_argument('--out', '-o', help='out name for acc and model')
args = parser.parse_args()

if __name__ == '__main__':
	torch.manual_seed(0)
	#data_dir = "./data_09"
	main(args.data, args.out, args.gpu)
