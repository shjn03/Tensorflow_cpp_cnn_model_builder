
#include "pch.h"
#include <iostream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"
#include "mnist.h"

using  namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;




int f() {
	static int i = 1;
	return i++;
};

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	std::vector<T> vec(first, last);
	return vec;
}

int main() {

	//setting params
	int num_classes = 10;
	int num_batchs = 32;
	int num_epochs = 10;
	int learning_rate = 0.0001;
	int image_height = 28;
	int image_width = 28;
	int image_channels = 1;
	int imageDim = image_height* image_width*image_channels;
	int num_trains = 60000;
	int num_iterations = (int)(num_trains/ num_batchs);
	int num_tests = 10000;


	Scope scope = Scope::NewRootScope();

	//入力,教師データのPlaceholder	
	auto y = Placeholder(scope, DT_FLOAT);

	//define CNN for mnist
	auto input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT, ops::Placeholder::Shape({ num_batchs, 28,28,1 }));
	Variable W1(scope, { 5,5,1,32 }, DT_FLOAT);
	Variable B1(scope, { 32 }, DT_FLOAT);
	auto x1_1 = ops::Conv2D(scope.WithOpName("conv1"), input, W1, { 1,1,1,1 }, "SAME");
	auto x1_2 = ops::BiasAdd(scope.WithOpName("add1"), x1_1, B1);
	auto x1_3 = ops::Relu(scope.WithOpName("relu1"), x1_2);
	auto x1_4 = ops::MaxPool(scope.WithOpName("maxPool1"), x1_3, { 1,2,2,1 }, { 1,2,2,1 }, "SAME");
	Variable W2(scope, { 5,5,32,64 }, DT_FLOAT);
	Variable B2(scope, { 64 }, DT_FLOAT);
	auto x2_1 = ops::Conv2D(scope.WithOpName("conv2"), x1_4, W2, { 1,1,1,1 }, "SAME");
	auto x2_2 = ops::Add(scope.WithOpName("add2"), x2_1, B2);
	auto x2_3 = ops::Relu(scope.WithOpName("relu1"), x2_2);
	auto x2_4 = ops::MaxPool(scope.WithOpName("maxPool2"), x2_3, { 1,2,2,1 }, { 1,2,2,1 }, "SAME");
	Variable W3(scope, { 7 * 7 * 64, 1024 }, DT_FLOAT);
	Variable B3(scope, { 1024 }, DT_FLOAT);
	auto x3_1 = ops::Reshape(scope.WithOpName("rehape1"), x2_4, { num_batchs, 7 * 7 * 64 });
	auto x3_2 = ops::BatchMatMul(scope.WithOpName("fc1"), x3_1, W3);
	auto x3_3 = ops::Add(scope.WithOpName("add3"), x3_2, B3);
	auto x3_4 = ops::Relu(scope.WithOpName("relu3"), x3_3);

	Variable W4(scope, { 1024, num_classes }, DT_FLOAT);
	Variable B4(scope, { num_classes }, DT_FLOAT);
	auto x4_1 = ops::BatchMatMul(scope.WithOpName("fc2"), x3_4, W4);
	auto x4_2 = ops::Add(scope.WithOpName("add4"), x4_1, B4);
	auto x4_3 = ops::Softmax(scope.WithOpName("softmax"), x4_2);

	auto output = ops::Placeholder(scope.WithOpName("output"), DT_FLOAT, ops::Placeholder::Shape({ num_batchs, num_classes }));

 	auto assign_w1 = Assign(scope, W1, RandomNormal(scope, { 5,5,1,32 }, DT_FLOAT));
	auto assign_w2 = Assign(scope, W2, RandomNormal(scope, { 5,5,32,64 }, DT_FLOAT));
	auto assign_w3 = Assign(scope, W3, RandomNormal(scope, { 7 * 7 * 64, 1024 }, DT_FLOAT));
	auto assign_w4 = Assign(scope, W4, RandomNormal(scope, { 1024, num_classes }, DT_FLOAT));

	auto assign_b1 = Assign(scope, B1, RandomNormal(scope, { 32 }, DT_FLOAT));
	auto assign_b2 = Assign(scope, B2, RandomNormal(scope, { 64 }, DT_FLOAT));
	auto assign_b3 = Assign(scope, B3, RandomNormal(scope, { 1024 }, DT_FLOAT));
	auto assign_b4 = Assign(scope, B4, RandomNormal(scope, { num_classes }, DT_FLOAT));

	//set loss function
	auto loss =ReduceSum(scope,  Mul(scope,x4_3, Log(scope,output)), { 0, 1 });
	//auto loss = ReduceMean(scope, Square(scope, Sub(scope,x4_3 , output)), { 0, 1 });
	//auto loss = SoftmaxCrossEntropyWithLogits(scope, x4_3, output);
	
	//appy gradient
	std::vector<Output> grad_outputs;
	TF_CHECK_OK(AddSymbolicGradients(scope, { loss }, { W1, W2, W3, W4 ,B1, B2, B3, B4 }, &grad_outputs));

	//apply GradientDesncent for each varibles
	auto apply_w1 = ApplyGradientDescent(scope, W1, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[0] });
	auto apply_w2 = ApplyGradientDescent(scope, W2, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[1] });
	auto apply_w3 = ApplyGradientDescent(scope, W3, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[2] });
	auto apply_w4 = ApplyGradientDescent(scope, W4, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[3] });
	
	auto apply_b1 = ApplyGradientDescent(scope, B1, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[4] });
	auto apply_b2 = ApplyGradientDescent(scope, B2, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[5] });
	auto apply_b3 = ApplyGradientDescent(scope, B3, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[6] });
	auto apply_b4 = ApplyGradientDescent(scope, B4, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[7] });


	cout << "preparing input data..." << endl;
	// trainning data(minibatch)
	Tensor x_batch(DT_FLOAT, TensorShape({ num_batchs, 28,28,1 }));
	Tensor y_batch(DT_FLOAT, TensorShape({ num_batchs, num_classes }));

	//test data
	Tensor x_test(DT_FLOAT, TensorShape({ num_tests, 28,28,1 }));
	Tensor y_test(DT_FLOAT, TensorShape({ num_tests, num_classes }));

	//load test data
	MNIST mnist = MNIST("../MNIST_data/");
	auto dst = x_test.flat<float>().data();
	auto dst2 = y_test.flat<float>().data();
	for (int i = 0; i < num_tests; i++) {
		auto img = mnist.testData.at(i).pixelData;
		std::copy_n(img.begin(), imageDim, dst);
		dst += imageDim;
		auto label = mnist.testData.at(i).output;
		std::copy_n(label.begin(), num_classes, dst2);
		dst2 += num_classes;
	}

	vector<int> train_idx(num_trains);
	vector<int> batch_idx(num_batchs);
	generate(train_idx.begin(), train_idx.end(), f);

	std::vector<Tensor> outputs;

	//create　session from scope
	ClientSession session(scope);

	//nititialize all varibles
	TF_CHECK_OK(session.Run({ assign_w1, assign_w2,assign_w3, assign_w4 ,assign_b1,  assign_b2,assign_b3,assign_b4 }, nullptr));

	random_device	rd;
	mt19937 g(rd());
	
	for (int e = 0; e < num_epochs; e++) {
		cout << "EPOCHS " << e << " start..." << endl;
		shuffle(train_idx.begin(), train_idx.begin() + num_trains, g);

		for (int i = 0; i <= num_iterations; i++) {
			//load train mini batch data
			int start = i * num_batchs;
			int end = i* num_batchs+num_batchs;
			batch_idx = slice(train_idx, start, end);

			auto dst = x_batch.flat<float>().data();
			auto dst2 = y_batch.flat<float>().data();
			for (int j : batch_idx) {
				auto img = mnist.trainingData.at(i).pixelData;
				std::copy_n(img.begin(), imageDim, dst);
				dst += imageDim;
				auto label = mnist.trainingData.at(i).output;
				std::copy_n(label.begin(), num_classes, dst2);
				dst2 += num_classes;
			}

			TF_CHECK_OK(session.Run({ {input, x_batch }, { output, y_batch } }, { loss }, &outputs));

			printf("epoch = %5d, iteraiton = % 5d, loss = %12.1f\n", e,i, outputs[0].scalar<float>()());
			//if (e == num_epochs) {
			//	break;
			//}

			//
			TF_CHECK_OK(session.Run({ {input,x_batch },{ output, y_batch } }, { apply_w1, apply_b1, apply_w2, apply_b2, apply_w3, apply_b3, apply_w4, apply_b4 }, nullptr));


		}
	}
}
