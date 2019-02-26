// cae_model_builder.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

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


#define printTensor(T, d) std::cout<< (T).tensor<float, (d)>() << std::endl

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
	BOOL DEBUG_NETWORK = FALSE;//foward prop only
	int num_classes = 10;
	int num_batchs = 32;
	int num_epochs = 10;
	int learning_rate = 0.0001;
	int image_height = 28;
	int image_width = 28;
	int image_channels = 1;
	int intermediate_dim = 256;
	int encode_dim = 16;
	int imageDim = image_height * image_width*image_channels;
	int num_trains = 60000;
	int num_iterations = (int)(num_trains / num_batchs);
	int num_tests = 10000;
	int encode_height = (int)image_height / 4;
	int encode_width = (int)image_width / 4;
	int num_filters1 = 32;
	int num_filters2 = 64;
	int filter_size = 3;

	Scope scope = Scope::NewRootScope();

	//入力,教師データのPlaceholder	
	auto y = Placeholder(scope, DT_FLOAT);

	//define CNN for mnist
	auto input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT, ops::Placeholder::Shape({ num_batchs, image_height,image_width,image_channels }));
	Variable W1(scope, { filter_size,filter_size,image_channels,num_filters1 }, DT_FLOAT);
	Variable B1(scope, { num_filters1 }, DT_FLOAT);
	auto x1_1 = ops::Conv2D(scope.WithOpName("conv1"), input, W1, { 1,1,1,1 }, "SAME");
	auto x1_2 = ops::BiasAdd(scope.WithOpName("add1"), x1_1, B1);
	auto x1_3 = ops::Relu(scope.WithOpName("relu1"), x1_2);
	auto x1_4 = ops::MaxPool(scope.WithOpName("maxPool1"), x1_3, { 1,2,2,1 }, { 1,2,2,1 }, "SAME");
	Variable W2(scope, { filter_size,filter_size,num_filters1,num_filters2 }, DT_FLOAT);
	Variable B2(scope, { num_filters2 }, DT_FLOAT);
	auto x2_1 = ops::Conv2D(scope.WithOpName("conv2"), x1_4, W2, { 1,1,1,1 }, "SAME");
	auto x2_2 = ops::Add(scope.WithOpName("add2"), x2_1, B2);
	auto x2_3 = ops::Relu(scope.WithOpName("relu2"), x2_2);
	auto x2_4 = ops::MaxPool(scope.WithOpName("maxPool2"), x2_3, { 1,2,2,1 }, { 1,2,2,1 }, "SAME");
	Variable W3(scope, { encode_height*encode_width*num_filters2, intermediate_dim }, DT_FLOAT);
	Variable B3(scope, { intermediate_dim }, DT_FLOAT);
	auto x3_1 = ops::Reshape(scope.WithOpName("reshape1"), x2_4, { num_batchs, encode_height*encode_width*num_filters2 });
	auto x3_2 = ops::BatchMatMul(scope.WithOpName("fc1"), x3_1, W3);
	auto x3_3 = ops::Add(scope.WithOpName("add3"), x3_2, B3);
	auto x3_4 = ops::Relu(scope.WithOpName("relu3"), x3_3);
	Variable W4(scope, { intermediate_dim ,encode_dim }, DT_FLOAT);
	Variable B4(scope, { encode_dim }, DT_FLOAT);
	auto x4_1 = ops::BatchMatMul(scope.WithOpName("fc2"), x3_4, W4);
	auto x4_2 = ops::Add(scope.WithOpName("add4"), x4_1, B4);
	auto x4_3 = ops::Relu(scope.WithOpName("relu4"), x4_2);
	//auto x4_3 = ops::Softmax(scope.WithOpName("softmax"), x4_2);
	Variable W5(scope, { encode_dim, encode_height*encode_width*num_filters2 }, DT_FLOAT);
	Variable B5(scope, { encode_height*encode_width*num_filters2 }, DT_FLOAT);
	auto x5_1 = ops::BatchMatMul(scope.WithOpName("fc3"), x4_3, W5);
	auto x5_2 = ops::Add(scope.WithOpName("add5"), x5_1, B5);
	auto x5_3 = ops::Relu(scope.WithOpName("relu5"), x5_2);
	Variable W6(scope, { filter_size,filter_size,num_filters2,num_filters1 }, DT_FLOAT);
	Variable B6(scope, { num_filters1 }, DT_FLOAT);
	auto x6_1 = ops::Reshape(scope.WithOpName("reshape2"), x5_3, { num_batchs,  encode_height,encode_width,num_filters2 });
	//Upsamplingの代替
	auto x6_2 = ops::ResizeNearestNeighbor(scope.WithOpName("maxunpool1"), x6_1, { encode_height * 2,encode_width * 2 });
	//kerasの実装は標準がnearestなので踏襲。もしかしたらResizeBilinearやResizeBicubicで精度若干変わる？
	//ただし(2,2)のupsampleingじゃ変わらんかも　Deeplabとか実装用？
	auto x6_3 = ops::Conv2D(scope.WithOpName("conv6"), x6_2, W6, { 1,1,1,1 }, "SAME");
	auto x6_4 = ops::Add(scope.WithOpName("add6"), x6_3, B6);
	auto x6_5 = ops::Relu(scope.WithOpName("relu6"), x6_4);

	Variable W7(scope, { filter_size,filter_size,num_filters1,image_channels }, DT_FLOAT);
	Variable B7(scope, { image_channels }, DT_FLOAT);
	//Upsamplingの代替
	auto x7_1 = ops::ResizeNearestNeighbor(scope.WithOpName("maxunpool2"), x6_5, { image_height,image_width });
	auto x7_2 = ops::Conv2D(scope.WithOpName("conv7"), x7_1, W7, { 1,1,1,1 }, "SAME");
	auto x7_3 = ops::Add(scope.WithOpName("add7"), x7_2, B7);
	auto x7_4 = ops::Sigmoid(scope.WithOpName("sigmoid"), x7_3);

	auto output = ops::Placeholder(scope.WithOpName("output"), DT_FLOAT, ops::Placeholder::Shape({ num_batchs,image_height,image_width,image_channels }));

	auto assign_w1 = Assign(scope, W1, RandomNormal(scope, { filter_size,filter_size,image_channels,num_filters1 }, DT_FLOAT));
	auto assign_w2 = Assign(scope, W2, RandomNormal(scope, { filter_size,filter_size,num_filters1,num_filters2 }, DT_FLOAT));
	auto assign_w3 = Assign(scope, W3, RandomNormal(scope, { encode_height*encode_width*num_filters2, intermediate_dim }, DT_FLOAT));
	auto assign_w4 = Assign(scope, W4, RandomNormal(scope, { intermediate_dim,encode_dim }, DT_FLOAT));
	auto assign_w5 = Assign(scope, W5, RandomNormal(scope, { encode_dim, encode_height*encode_width*num_filters2 }, DT_FLOAT));
	auto assign_w6 = Assign(scope, W6, RandomNormal(scope, { filter_size,filter_size,num_filters2,num_filters1 }, DT_FLOAT));
	auto assign_w7 = Assign(scope, W7, RandomNormal(scope, { filter_size,filter_size,num_filters1,image_channels }, DT_FLOAT));

	auto assign_b1 = Assign(scope, B1, RandomNormal(scope, { num_filters1 }, DT_FLOAT));
	auto assign_b2 = Assign(scope, B2, RandomNormal(scope, { num_filters2 }, DT_FLOAT));
	auto assign_b3 = Assign(scope, B3, RandomNormal(scope, { intermediate_dim }, DT_FLOAT));
	auto assign_b4 = Assign(scope, B4, RandomNormal(scope, { encode_dim }, DT_FLOAT));
	auto assign_b5 = Assign(scope, B5, RandomNormal(scope, { encode_height*encode_width*num_filters2 }, DT_FLOAT));
	auto assign_b6 = Assign(scope, B6, RandomNormal(scope, { num_filters1 }, DT_FLOAT));
	auto assign_b7 = Assign(scope, B7, RandomNormal(scope, { image_channels }, DT_FLOAT));
	
	//set loss function
	//auto loss = ReduceSum(scope, Mul(scope, x4_3, Log(scope, output)), { 0, 1 });
	auto loss = ReduceMean(scope, Square(scope, Sub(scope, x7_4, output)), { 0, 1 });
	//auto loss = SoftmaxCrossEntropyWithLogits(scope, x4_3, output);
	//apply GradientDesncent for each varibles
	
	if (DEBUG_NETWORK) {
		//appy gradient
		std::vector<Output> grad_outputs;
		TF_CHECK_OK(AddSymbolicGradients(scope, { loss }, { W1, W2, W3, W4 ,W5, W6, W7,B1, B2, B3, B4,B5, B6, B7 }, &grad_outputs));

		//apply GradientDesncent for each varibles
		auto apply_w1 = ApplyGradientDescent(scope, W1, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[0] });
		auto apply_w2 = ApplyGradientDescent(scope, W2, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[1] });
		auto apply_w3 = ApplyGradientDescent(scope, W3, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[2] });
		auto apply_w4 = ApplyGradientDescent(scope, W4, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[3] });
		auto apply_w5 = ApplyGradientDescent(scope, W5, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[4] });
		auto apply_w6 = ApplyGradientDescent(scope, W6, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[5] });
		auto apply_w7 = ApplyGradientDescent(scope, W7, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[6] });

		auto apply_b1 = ApplyGradientDescent(scope, B1, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[7] });
		auto apply_b2 = ApplyGradientDescent(scope, B2, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[8] });
		auto apply_b3 = ApplyGradientDescent(scope, B3, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[9] });
		auto apply_b4 = ApplyGradientDescent(scope, B4, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[10] });
		auto apply_b5 = ApplyGradientDescent(scope, B5, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[11] });
		auto apply_b6 = ApplyGradientDescent(scope, B6, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[12] });
		auto apply_b7 = ApplyGradientDescent(scope, B7, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[13] });
	}

	cout << "preparing input data..." << endl;
	// trainning data(minibatch)
	Tensor x_batch(DT_FLOAT, TensorShape({ num_batchs, image_height,image_width,image_channels }));
	Tensor y_batch(DT_FLOAT, TensorShape({ num_batchs, image_height,image_width,image_channels }));

	//test data
	Tensor x_test(DT_FLOAT, TensorShape({ num_tests,image_height,image_width,image_channels }));
	Tensor y_test(DT_FLOAT, TensorShape({ num_tests, image_height,image_width,image_channels }));

	//load test data
	MNIST mnist = MNIST("../MNIST_data/");
	auto dst = x_test.flat<float>().data();
	auto dst2 = y_test.flat<float>().data();
	for (int i = 0; i < num_tests; i++) {
		auto img = mnist.testData.at(i).pixelData;
		std::copy_n(img.begin(), imageDim, dst);
		dst += imageDim;
		std::copy_n(img.begin(), imageDim, dst2);
		dst2 += imageDim;
	}

	vector<int> train_idx(num_trains);
	vector<int> batch_idx(num_batchs);
	generate(train_idx.begin(), train_idx.end(), f);

	std::vector<Tensor> outputs;

	//create　session from scope
	ClientSession session(scope);

	//nititialize all varibles
	TF_CHECK_OK(session.Run({ assign_w1, assign_w2,assign_w3, assign_w4, assign_w5,assign_w6, assign_w7,assign_b1, assign_b2,assign_b3,assign_b4,assign_b5,assign_b6,assign_b7 }, nullptr));
	//TF_CHECK_OK(session.Run({ assign_w1 }, nullptr));
	//TF_CHECK_OK(session.Run({ assign_b1, assign_b2,assign_b3,assign_b4,assign_b5,assign_b6,assign_b7 }, nullptr));

	random_device	rd;
	mt19937 g(rd());

	for (int e = 0; e < num_epochs; e++) {
		cout << "EPOCHS " << e << " start..." << endl;
		shuffle(train_idx.begin(), train_idx.begin() + num_trains, g);

		for (int i = 0; i <= num_iterations; i++) {
			//load train mini batch data
			int start = i * num_batchs;
			int end = i * num_batchs + num_batchs;
			batch_idx = slice(train_idx, start, end);

			auto dst = x_batch.flat<float>().data();
			auto dst2 = y_batch.flat<float>().data();
			for (int j : batch_idx) {
				auto img = mnist.trainingData.at(i).pixelData;
				std::copy_n(img.begin(), imageDim, dst);
				dst += imageDim;
				std::copy_n(img.begin(), imageDim, dst2);
				dst2 += imageDim;
			}

			TF_CHECK_OK(session.Run({ {input, x_batch }, { output, y_batch } }, { loss }, &outputs));

			printf("epoch = %5d, iteraiton = % 5d, loss = %12.1f\n", e, i, outputs[0].scalar<float>()());
			//if (e == num_epochs) {
			//	break;
			//}

			//
			//if (DEBUG_NETWORK) {
			//	TF_CHECK_OK(session.Run({ {input,x_batch },{ output, y_batch } }, { apply_w1, apply_b1, apply_w2, apply_b2, apply_w3, apply_b3, apply_w4, apply_b4,apply_w5, apply_b5,apply_w6, apply_b6 , apply_w7, apply_b7 }, nullptr));
			//}

		}
	}
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
