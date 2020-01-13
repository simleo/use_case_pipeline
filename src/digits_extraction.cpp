#define _USE_MATH_DEFINES
#include "ecvl/core.h"
#include "ecvl/eddl.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <queue>

using namespace ecvl;
using namespace std;
using namespace std::filesystem;

void FindMarkers(const Image& src, Image& dst, int radius)
{

    Image out({ src.dims_[0] - 1, src.dims_[1] - 1, src.dims_[2] }, DataType::int32, src.channels_, ColorType::GRAY);
    memset(out.data_, 0, out.datasize_);

    size_t size = radius * 2;

    ConstContiguousViewXYC<DataType::float64> vsrc(src);
    ContiguousViewXYC<DataType::int32> vdst(out);

    for (int y = radius; y < vdst.height() - radius; ++y) {
        for (int x = radius; x < vdst.width() - radius; ++x) {
            float all = vsrc(x + radius, y + radius, 0) - vsrc(x - radius, y + radius, 0) - vsrc(x + radius, y - radius, 0) + vsrc(x - radius, y - radius, 0);
            float sqtl = vsrc(x, y, 0) - vsrc(x - radius, y, 0) - vsrc(x, y - radius, 0) + vsrc(x - radius, y - radius, 0);
            float sqbr = vsrc(x + radius, y + radius, 0) - vsrc(x, y + radius, 0) - vsrc(x + radius, y, 0) + vsrc(x, y, 0);

            vdst(x, y, 0) = static_cast<int>(abs(all - 2 * sqtl - 2 * sqbr));
        }
    }

    dst = move(out);
}

void CutOnMarkers(const Image& src, Image& dst,
    const vector<pair<ecvl::Point2i, ecvl::Point2i>>& maxs,
    vector<int> dst_dims = {})
{
    if (dst_dims.empty()) {
        dst_dims = src.dims_;
    }
    Image out(dst_dims, DataType::uint8, "xyc", ColorType::GRAY);
    ConstContiguousViewXYC<DataType::uint8> vsrc(src);
    ContiguousViewXYC<DataType::uint8> vout(out);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(Eigen::Index(maxs.size() * 2), Eigen::Index(6));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(Eigen::Index(maxs.size() * 2), Eigen::Index(1));

    for (int r = 0; r < maxs.size() * 2; r += 2) {
        A(r, 0) = maxs[r / 2].second[0];
        A(r, 1) = maxs[r / 2].second[1];
        A(r, 2) = 0;
        A(r, 3) = 0;
        A(r, 4) = 1;
        A(r, 5) = 0;
        b(r, 0) = maxs[r / 2].first[0];

        A(r + 1, 0) = 0;
        A(r + 1, 1) = 0;
        A(r + 1, 2) = maxs[r / 2].second[0];
        A(r + 1, 3) = maxs[r / 2].second[1];
        A(r + 1, 4) = 0;
        A(r + 1, 5) = 1;
        b(r + 1, 0) = maxs[r / 2].first[1];
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ainv = A.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> z = Ainv * b;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rt_point(3, 3);
    rt_point(0, 0) = z(0, 0);
    rt_point(0, 1) = z(1, 0);
    rt_point(1, 0) = z(2, 0);
    rt_point(1, 1) = z(3, 0);
    rt_point(0, 2) = z(4, 0);
    rt_point(1, 2) = z(5, 0);
    rt_point(2, 0) = 0;
    rt_point(2, 1) = 0;
    rt_point(2, 2) = 1;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cur_coord(3, 1);
    for (size_t y = 0; y < vout.height(); ++y) {
        for (size_t x = 0; x < vout.width(); ++x) {
            cur_coord(0, 0) = x;
            cur_coord(1, 0) = y;
            cur_coord(2, 0) = 1;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rt_coord = rt_point * cur_coord;

            int c = floor(rt_coord(0, 0));
            int r = floor(rt_coord(1, 0));
            int32_t cur_value = 0;
            if (!(c < 0 || c + 1 >= vsrc.width() || r < 0 || r + 1 >= vsrc.height())) {

                double a = rt_coord(0, 0) - c;
                double b = rt_coord(1, 0) - r;

                cur_value = vsrc(c, r, 0)   * (1 - a)*(1 - b) +
                    vsrc(c + 1, r, 0)       * a*(1 - b) +
                    vsrc(c, r + 1, 0)       * (1 - a)*b +
                    vsrc(c + 1, r + 1, 0)   * a*b;
            }

            vout(x, y, 0) = cur_value;

        }
    }
    dst = std::move(out);
}

int main()
{
    vector<string> names{ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "plus", "dash", "e_lower", "e_upper", "dot", "comma", "black" };
    vector<int> index(names.size(), 0);

    path input_path("imgs");
    path output_path("../extracted_data");

    for (int c = 0; c < names.size(); ++c) {
        create_directories(output_path / path(names[c]));
    }

    double scale = 1;
    int acquired_square_size = 240;
    acquired_square_size *= scale;

    for (auto& p : directory_iterator(input_path)) {
        Image src, dst;
        if (!ImRead(p, src)) {
            return EXIT_FAILURE;
        }
        ChangeColorSpace(src, src, ColorType::GRAY);
        ResizeScale(src, src, { scale,scale });

        double maxval = 255;
        Image integral_img = src;
        IntegralImage(integral_img, integral_img);

        Image score;
        FindMarkers(integral_img, score, acquired_square_size / 2);
        NonMaximaSuppression(score, score);

        const size_t n = 5;
        vector<Point2i> maxs = GetMaxN(score, n);

        // Sum of distances squared all with all
        vector<pair<double, Point2i>> distances(n, make_pair(0.0, Point2i{ -1,-1 }));
        for (size_t i = 0; i < n; ++i) {
            distances[i].second = maxs[i];
            for (size_t j = i + 1; j < n; ++j) {
                double d = SqDist(maxs[i], maxs[j]);
                distances[i].first += d;
                distances[j].first += d;
            }
        }
        sort(begin(distances), end(distances));

        // Ptl  Ptc  Ptr
        //
        // Pbl       Pbr

        // Ptc is the minimum
        Point2i Ptc = distances[0].second;
        // Ptl e Ptr are the next two (then check whether to swap them)
        Point2i Ptl = distances[1].second;
        Point2i Ptr = distances[2].second;
        // Pbl e Pbr are the next two (Pbl is the nearest to Ptl)
        Point2i Pbl = distances[3].second;
        Point2i Pbr = distances[4].second;
        if (SqDist(Ptl, Pbl) > SqDist(Ptl, Pbr)) {
            swap(Pbl, Pbr);
        }

        // Check if vector Vlr = Ptl->Ptr keep to the right the vector Vtb = Ptl->Pbl
        Point2i Vlr{ Ptr[0] - Ptl[0],Ptr[1] - Ptl[1] };
        Point2i Vtb{ Pbl[0] - Ptl[0],Pbl[1] - Ptl[1] };
        if (Vlr[0] * Vtb[1] - Vlr[1] * Vtb[0] < 0) {
            swap(Ptl, Ptr);
            swap(Pbl, Pbr);
        }

        int square_size = 120 * scale; //desired;
        int desired_width = 36 * square_size;
        int desired_height = 52 * square_size;

        vector<pair<Point2i, Point2i>> corrisp{
            {Ptl, {1 * square_size, 1 * square_size}},
            {Ptc, {18 * square_size, 1 * square_size}},
            {Ptr, {35 * square_size, 1 * square_size}},
            {Pbl, {1 * square_size, 51 * square_size}},
            {Pbr, {35 * square_size, 51 * square_size}},
        };

        CutOnMarkers(src, dst, corrisp, { desired_width, desired_height, 1 });

        int first_row = 5;
        int last_row = 25;
        vector<int> init{ square_size , square_size * first_row }; // x, y
        vector<int> step{ 2 * square_size, 2 * square_size };      // x, y
        vector<int> block_size{ square_size, square_size, -1 };    // x, y, z (for the view)

        vector<int> y_coords;
        y_coords.reserve(10);
        int c = square_size * 3 / 2;

        ConstContiguousViewXYC<DataType::uint8> vdst(dst);
        for (int r = init[1] - square_size; r < 25 * square_size; ++r) {
            if (vdst(c, r, 0) < 128) {
                y_coords.push_back(r);
                r += square_size * 3 / 2;
            }
        }

        for (int r = 0; r < (int)y_coords.size(); ++r) {
            for (int c = 0; c < 17; ++c) {
                vector<int> cur_pos{ init[0] + step[0] * c, y_coords[r], 0 };
                View<DataType::uint8> v(dst, cur_pos, block_size);
                string name = to_string(index[c]++);
                name = names[c] + "_" + std::string(6 - name.length(), '0') + name + ".png";
                path cur_path = output_path / path(names[c]) / path(name);

                ImWrite(cur_path, v);
            }
        }
        cout << "Image " << p.path().string() << " done" << endl;
    }

    return 0;
}