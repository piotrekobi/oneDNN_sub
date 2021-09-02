#include <../examples/example_utils.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include <iostream>
#include <random>

void random_fill_vector(int seed, int size, std::vector<float> &vec)
{
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> distr(0, 256);
    vec.resize(size);
    std::generate(vec.begin(), vec.end(), [&distr, &eng]() { return distr(eng); });
}

std::vector<float> manual_sub(std::vector<float> &vec, float value)
{
    std::vector<float> sub_vec(vec.size());
    std::generate(sub_vec.begin(), sub_vec.end(), [&, i = 0]() mutable { return vec[i++] - value; });
    return sub_vec;
}

int main()
{
    const int batch_dim = 1, channel_dim = 10, width = 5, height = 3;
    const int data_size = batch_dim * channel_dim * width * height;
    const int random_seed = 1337;
    const float sub_value = 10.0;

    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream eng_stream(eng);

    std::vector<float> data;
    random_fill_vector(random_seed, data_size, data);

    auto memory_descriptor = dnnl::memory::desc({batch_dim, channel_dim, height, width}, dnnl::memory::data_type::f32,
                                                dnnl::memory::format_tag::nhwc);

    auto src_memory_object = dnnl::memory(memory_descriptor, eng);

    write_to_dnnl_memory(data.data(), src_memory_object);

    auto dst_memory_object = dnnl::memory(memory_descriptor, eng);

    auto sub_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_linear,
                                             memory_descriptor, 1, -sub_value);

    auto sub_pd = dnnl::eltwise_forward::primitive_desc(sub_d, eng);

    auto sub = dnnl::eltwise_forward(sub_pd);

    sub.execute(eng_stream, {{DNNL_ARG_SRC, src_memory_object}, {DNNL_ARG_DST, dst_memory_object}});
    eng_stream.wait();

    std::vector<float> sub_data(data_size);
    read_from_dnnl_memory(sub_data.data(), dst_memory_object);

    std::vector<float> manual_sub_data = manual_sub(data, sub_value);

    for (int i = 0; i < data.size(); i++)
    {
        std::cout << data[i] << " " << sub_data[i] << " " << manual_sub_data[i] << std::endl;
    }

    return 0;
}
