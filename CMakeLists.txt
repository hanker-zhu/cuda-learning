cmake_minimum_required(VERSION 3.8)
project(cuda-learning LANGUAGES CUDA)

file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

foreach(cuda_file ${CUDA_SOURCES})
    # 获取相对路径（去掉src/前缀）
    file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}/src" "${cuda_file}")
    get_filename_component(target_dir "${rel_path}" DIRECTORY)
    get_filename_component(file_we "${rel_path}" NAME_WE)

    # 构造输出目录（build/ + 相对目录）
    set(output_dir "${CMAKE_BINARY_DIR}/build/${target_dir}")
    file(MAKE_DIRECTORY "${output_dir}")

    # 构造输出文件名
    set(output_file "${output_dir}/${file_we}")

    # 添加自定义命令，直接用nvcc编译
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND nvcc ${cuda_file} -o ${output_file}
        DEPENDS ${cuda_file}
        COMMENT "Building ${output_file} from ${cuda_file}"
    )
    add_custom_target(${target_dir}_${file_we}_target ALL DEPENDS ${output_file})
endforeach()