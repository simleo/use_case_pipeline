#include "ecvl/core.h"
#include "ecvl/eddl.h"
#include "ecvl/dataset_parser.h"
#include "models/models.h"
#include "utils/utils.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>

using namespace ecvl;
using namespace eddl;
using namespace std;
using namespace std::filesystem;

int main()
{
    // Settings
    int epochs = 20;
    int batch_size = 12;
    std::vector<int> size{ 120,120 }; // Size of images
    DLDataset d("charset/charset.yml", batch_size, size);
    int num_classes = d.classes_.size();

    std::random_device rd;
    std::mt19937 g(rd());

    // Define network
    layer in = Input({ 3, size[0],  size[1] });
    layer out = LeNet(in, num_classes);
    model net = Model({ in }, { out });

    // Build model
    build(net,
        sgd(0.001, 0.9), // Optimizer
        { "soft_cross_entropy" }, // Losses
        { "categorical_accuracy" } // Metrics
    );

    //toGPU(net);

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Read the dataset
    cout << "Reading dataset" << endl;

    path output_path("../output_images");
    ofstream of("output.txt");

    // Prepare tensors which store batch
    tensor x = eddlT::create({ batch_size, d.n_channels_, size[0], size[1] });
    tensor y = eddlT::create({ batch_size, static_cast<int>(d.classes_.size()) });
    tensor output;
    tensor result;
    tensor target;
    tensor single_image;

    int num_samples = d.GetSplit().size();
    int num_batches = num_samples / batch_size;

    d.SetSplit("test");
    int num_samples_test = d.GetSplit().size();
    int num_batches_test = num_samples_test / batch_size;

    vector<int> indices(batch_size);
    iota(indices.begin(), indices.end(), 0);
    View<DataType::float32> img_t;
    bool save_images = true;
    vector<float> total_metric;
    Metric* m = getMetric("categorical_accuracy");

    cout << "Starting training" << endl;
    for (int i = 0; i < epochs; ++i) {
        auto current_path{ output_path / path("Epoch_" + to_string(i)) };

        for (int c = 0; c < d.classes_.size(); ++c) {
            create_directories(current_path / path(d.classes_[c]));
        }
        d.SetSplit("training");
        // Reset errors
        reset_loss(net);

        // Shuffle training list
        shuffle(std::begin(d.GetSplit()), std::end(d.GetSplit()), g);
        d.ResetAllBatches();

        // Feed batches to the model
        for (int j = 0; j < num_batches; ++j) {
            cout << "Epoch " << i << "/" << epochs << " (batch " << j + 1 << "/" << num_batches << ") - ";

            // Load a batch
            d.LoadBatch(x, y);

            // Preprocessing
            x->div_(255.0);

            // Prepare data
            vtensor tx{ x };
            vtensor ty{ y };

            // Train batch
            train_batch(net, tx, ty, indices);

            // Print errors
            print_loss(net, j);
            cout << endl;
        }

        // Evaluation
        d.SetSplit("test");
        total_metric.clear();

        cout << "Evaluate test:" << endl;
        for (int j = 0; j < num_batches_test; ++j) {
            // Load a batch
            d.LoadBatch(x, y);

            // Preprocessing
            x->div_(255.0);

            // Evaluate batch
            evaluate(net, { x }, { y });

            output = getTensor(out);

            for (int k = 0; k < batch_size; ++k) {
                result = eddlT::select(output, k);
                target = eddlT::select(y, k);

                total_metric.push_back(m->value(target, result));

                float max = std::numeric_limits<float>::min();
                int classe = -1;
                for (int i = 0; i < result->size; ++i) {
                    if (result->ptr[i] > max) {
                        max = result->ptr[i];
                        classe = i;
                    }
                }

                single_image = eddlT::select(x, k);
                TensorToView(single_image, img_t);
                single_image->mult_(255.);

                if (save_images) {
                    ImageSqueeze(img_t);
                    path cur_path = current_path / d.classes_[classe] / path("batch_" + to_string(j) + "_img_" + to_string(k) + ".png");
                    ImWrite(cur_path, img_t);
                }
                delete result;
                delete target;
                delete single_image;
            }
        }

        float total_avg = accumulate(total_metric.begin(), total_metric.end(), 0.0) / total_metric.size();
        cout << "Epoch " << i << " - total categorical accuracy: " << total_avg << endl;
        of << "Epoch " << i << " - total categorical accuracy: " << total_avg << "\n";

        cout << "Saving weights..." << endl;
        save(net, "charset_classification_checkpoint_epoch_" + to_string(i) + ".bin", "bin");
    }

    of.close();
    delete x;
    delete y;
    delete output;

    //_CrtDumpMemoryLeaks();
    return EXIT_SUCCESS;
}