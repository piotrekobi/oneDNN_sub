#include <../examples/example_utils.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include <iostream>
#include <random>

template <typename T> std::vector<T> generate_random_vector(int seed, int size)
{
    std::vector<T> vec(size);

    std::default_random_engine eng(seed);
    std::uniform_real_distribution<T> distr(0, 128);

    std::generate(vec.begin(), vec.end(), [&distr, &eng]() { return distr(eng); });
    return vec;
}

template <typename T> std::vector<T> manual_sub(std::vector<T> &vec, const T value)
{
    std::vector<T> sub_vec(vec.size());
    int i = 0;
    std::generate(sub_vec.begin(), sub_vec.end(), [&, i]() mutable { return vec[i++] - value; });
    return sub_vec;
}

template <typename T>
std::vector<T> dnnl_sub(std::vector<T> &vec, const T value, const dnnl::memory::data_type data_type,
                        const dnnl::memory::format_tag format_tag, const dnnl::memory::dims &dimensions,
                        const int data_size)
{
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream eng_stream(eng);

    dnnl::memory::format_tag::nwc;

    auto memory_descriptor = dnnl::memory::desc(dimensions, data_type, format_tag);

    auto src_memory_object = dnnl::memory(memory_descriptor, eng);

    write_to_dnnl_memory(vec.data(), src_memory_object);

    auto dst_memory_object = dnnl::memory(memory_descriptor, eng);

    auto sub_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_linear,
                                             memory_descriptor, 1, -value);

    auto sub_pd = dnnl::eltwise_forward::primitive_desc(sub_d, eng);

    auto sub = dnnl::eltwise_forward(sub_pd);

    sub.execute(eng_stream, {{DNNL_ARG_SRC, src_memory_object}, {DNNL_ARG_DST, dst_memory_object}});
    eng_stream.wait();

    std::vector<T> sub_data(data_size);
    read_from_dnnl_memory(sub_data.data(), dst_memory_object);
    return sub_data;
}

template <typename T> bool fp_compare(const T &A, const T &B)
{
    return abs(A - B) < 0.00001;
}

template <typename T>
bool test_dnnl_sub(const dnnl::memory::dims dimensions, const int random_seed, const T sub_value,
                   dnnl::memory::data_type data_type, dnnl::memory::format_tag format_tag)
{
    int data_size = 1;
    for (const int &dimension : dimensions)
        data_size *= dimension;

    std::vector<T> data = generate_random_vector<T>(random_seed, data_size);

    std::vector<T> dnnl_sub_data = dnnl_sub<T>(data, sub_value, data_type, format_tag, dimensions, data_size);
    std::vector<T> manual_sub_data = manual_sub<T>(data, sub_value);

    return std::equal(dnnl_sub_data.begin(), dnnl_sub_data.end(), manual_sub_data.begin(), fp_compare<T>);
}

std::string test_result(bool result)
{
    if (result)
        return "Passed";
    return "Failed";
}

int main()
{

    std::cout << "3 Dimension test: "
              << test_result(test_dnnl_sub<float>({1, 3, 5}, 0, 4.5, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::nwc))
              << std::endl;

    std::cout << "4 Dimension test: "
              << test_result(test_dnnl_sub<float>({1, 3, 3, 7}, 1337, 10, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::nhwc))
              << std::endl;

    std::cout << "5 Dimension test: "
              << test_result(test_dnnl_sub<float>({1, 2, 3, 4, 5}, 500, 11.5, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::ndhwc))
              << std::endl;

    return 0;
}
