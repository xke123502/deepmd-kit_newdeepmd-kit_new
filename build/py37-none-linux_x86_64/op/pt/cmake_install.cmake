# Install script for directory: /aisi/mnt/data_nas/jwzhou/deepmd-kit/source/op/pt

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/tmp/tmpdoyj9gdr/wheel/platlib")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deepmd/lib/libdeepmd_op_pt.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deepmd/lib/libdeepmd_op_pt.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deepmd/lib/libdeepmd_op_pt.so"
         RPATH "\$ORIGIN")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/deepmd/lib" TYPE MODULE FILES "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/op/pt/libdeepmd_op_pt.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deepmd/lib/libdeepmd_op_pt.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deepmd/lib/libdeepmd_op_pt.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deepmd/lib/libdeepmd_op_pt.so"
         OLD_RPATH "/aisi/mnt/data_nas/jwzhou/conda_env/mad/lib/python3.11/site-packages/torch/lib/../../torch.libs:/opt/intel/oneapi/mkl/2022.0.2/lib:/opt/intel/oneapi/mkl/2022.0.2/lib/intel64_win:/opt/intel/oneapi/mkl/2022.0.2/lib/win-x64:/aisi/mnt/data_nas/jwzhou/conda_env/mad/lib/python3.11/site-packages/torch/lib:/usr/local/cuda/lib64:/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/lib:"
         NEW_RPATH "\$ORIGIN")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deepmd/lib/libdeepmd_op_pt.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/op/pt/CMakeFiles/deepmd_op_pt.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

