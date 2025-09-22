set (LMP_INSTALL_PREFIX "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/USER-DEEPMD")
file(READ "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/lmp/lammps_install_list.txt" files)
string(REGEX REPLACE "\n" "" files "${files}")

foreach (cur_file ${files})
  file (
    INSTALL DESTINATION "${LMP_INSTALL_PREFIX}"
    USE_SOURCE_PERMISSIONS
    TYPE FILE
    FILES "${cur_file}"
    )
endforeach ()

file (
  INSTALL DESTINATION "${LMP_INSTALL_PREFIX}"
  TYPE FILE
  FILES "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/lmp/env.sh"
)

file (
  INSTALL DESTINATION "${LMP_INSTALL_PREFIX}"
  TYPE FILE
  FILES "/aisi/mnt/data_nas/jwzhou/deepmd-kit/build/py37-none-linux_x86_64/lmp/deepmd_version.h"
)
