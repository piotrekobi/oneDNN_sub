#include <../examples/example_utils.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include <iostream>
#include <random>

template <typename T> void random_fill_vector(int seed, int size, std::vector<T> &vec)
{
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<T> distr(0, 256);
    vec.resize(size);
    std::generate(vec.begin(), vec.end(), [&distr, &eng]() { return distr(eng); });
}

template <typename T> std::vector<T> manual_sub(std::vector<T> &vec, const T value)
{
    std::vector<T> sub_vec(vec.size());
    std::generate(sub_vec.begin(), sub_vec.end(), [&, i = 0]() mutable { return vec[i++] - value; });
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

int main()
{
    const int batch_dim = 1, channel_dim = 10, width = 5, height = 3;
    const int data_size = batch_dim * channel_dim * width * height;
    const int random_seed = 1337;

    const float sub_value = 10.0;

    const dnnl::memory::dims dimensions = {batch_dim, channel_dim, height, width};

    std::vector<float> data;
    random_fill_vector<float>(random_seed, data_size, data);

    std::vector<float> dnnl_sub_data = dnnl_sub<float>(data, sub_value, dnnl::memory::data_type::f32,
                                                       dnnl::memory::format_tag::nhwc, dimensions, data_size);
    std::vector<float> manual_sub_data = manual_sub<float>(data, sub_value);

    for (int i = 0; i < data.size(); i++)
    {
        std::cout << data[i] << " " << dnnl_sub_data[i] << " " << manual_sub_data[i] << std::endl;
    }

    return 0;
}
