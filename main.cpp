#include <../examples/example_utils.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include <iostream>
#include <random>

template <typename T> std::vector<T> generate_random_vector(int seed, int size, int min, int max)
{
    std::vector<T> vec(size);

    std::default_random_engine eng(seed);

    std::uniform_int_distribution<T> distr(min, max);

    std::generate(vec.begin(), vec.end(), [&distr, &eng]() { return distr(eng); });
    return vec;
}

template <> std::vector<float> generate_random_vector(int seed, int size, int min, int max)
{
    std::vector<float> vec(size);

    std::default_random_engine eng(seed);

    std::uniform_real_distribution<float> distr(0, 128);

    std::generate(vec.begin(), vec.end(), [&distr, &eng]() { return distr(eng); });

    return vec;
}

template <typename T> std::vector<T> manual_sub(const std::vector<T> &vec1, const std::vector<T> &vec2)
{
    std::vector<T> sub_vec(vec1.size());
    int i = -1;
    std::generate(sub_vec.begin(), sub_vec.end(), [&, i]() mutable {
        i++;
        return vec1[i] - vec2[i];
    });
    return sub_vec;
}

template <typename T>
std::vector<T> dnnl_sub(std::vector<T> &vec1, std::vector<T> &vec2, const dnnl::memory::data_type data_type,
                        const dnnl::memory::format_tag format_tag, const dnnl::memory::dims &dimensions,
                        const int data_size)
{
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream eng_stream(eng);

    auto memory_descriptor_0 = dnnl::memory::desc(dimensions, data_type, format_tag);
    auto memory_descriptor_1 = dnnl::memory::desc(dimensions, data_type, format_tag);
    auto memory_descriptor_dst = dnnl::memory::desc(dimensions, data_type, format_tag);

    auto src_memory_object_0 = dnnl::memory(memory_descriptor_0, eng);
    auto src_memory_object_1 = dnnl::memory(memory_descriptor_1, eng);
    auto dst_memory_object = dnnl::memory(memory_descriptor_dst, eng);

    write_to_dnnl_memory(vec1.data(), src_memory_object_0);
    write_to_dnnl_memory(vec2.data(), src_memory_object_1);

    auto sub_d = dnnl::binary::desc(dnnl::algorithm::binary_sub, memory_descriptor_0, memory_descriptor_1,
                                    memory_descriptor_dst);
    dnnl::primitive_attr binary_attr;
    auto sub_pd = dnnl::binary::primitive_desc(sub_d, eng);
    auto sub = dnnl::binary(sub_pd);

    sub.execute(eng_stream, {{DNNL_ARG_SRC_0, src_memory_object_0},
                             {DNNL_ARG_SRC_1, src_memory_object_1},
                             {DNNL_ARG_DST, dst_memory_object}});
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

    std::vector<T> data_1 = generate_random_vector<T>(random_seed, data_size, 50, 100);
    std::vector<T> data_2 = generate_random_vector<T>(random_seed + 5, data_size, 10, 20);

    std::vector<T> dnnl_sub_data = dnnl_sub<T>(data_1, data_2, data_type, format_tag, dimensions, data_size);
    std::vector<T> manual_sub_data = manual_sub<T>(data_1, data_2);

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

    std::cout << "3 Dimension float test: "
              << test_result(test_dnnl_sub<float>({1, 3, 5}, 0, 4.5, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::nwc))
              << std::endl;

    std::cout << "4 Dimension float test: "
              << test_result(test_dnnl_sub<float>({1, 3, 3, 7}, 1337, 10, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::nhwc))
              << std::endl;

    std::cout << "5 Dimension float test: "
              << test_result(test_dnnl_sub<float>({1, 2, 3, 4, 5}, 500, 11.5, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::ndhwc))
              << std::endl;

    std::cout << "3 Dimension char (int8) test: "
              << test_result(
                     test_dnnl_sub<char>({1, 3, 5}, 0, 5, dnnl::memory::data_type::s8, dnnl::memory::format_tag::nwc))
              << std::endl;

    std::cout << "4 Dimension char (int8) test: "
              << test_result(test_dnnl_sub<char>({1, 3, 3, 7}, 1337, 10, dnnl::memory::data_type::s8,
                                                 dnnl::memory::format_tag::nhwc))
              << std::endl;

    std::cout << "5 Dimension char (int8) test: "
              << test_result(test_dnnl_sub<char>({1, 2, 3, 4, 5}, 500, 12, dnnl::memory::data_type::s8,
                                                 dnnl::memory::format_tag::ndhwc))
              << std::endl;

    std::cout << "3 Dimension unsigned char test: "
              << test_result(test_dnnl_sub<unsigned char>({1, 3, 5}, 0, 5, dnnl::memory::data_type::u8,
                                                          dnnl::memory::format_tag::nwc))
              << std::endl;

    std::cout << "4 Dimension unsigned char test: "
              << test_result(test_dnnl_sub<unsigned char>({1, 3, 3, 7}, 1337, 10, dnnl::memory::data_type::u8,
                                                          dnnl::memory::format_tag::nhwc))
              << std::endl;

    std::cout << "5 Dimension unsigned char test: "
              << test_result(test_dnnl_sub<unsigned char>({1, 2, 3, 4, 5}, 500, 12, dnnl::memory::data_type::u8,
                                                          dnnl::memory::format_tag::ndhwc))
              << std::endl;

    return 0;
}