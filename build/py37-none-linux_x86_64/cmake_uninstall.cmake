if(NOT EXISTS "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: /aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/install_manifest.txt")
endif(NOT EXISTS "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/install_manifest.txt")

file(READ "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    exec_program(
      "/root/cmake-3.29.2-linux-x86_64/bin/cmake" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
      )
    if(NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    endif(NOT "${rm_retval}" STREQUAL 0)
  else(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
endforeach(file)
