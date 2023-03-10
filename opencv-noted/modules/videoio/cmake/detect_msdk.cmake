set(MFX_DEFS "")

if(NOT HAVE_MFX)
  find_package(VPL QUIET)
  if(VPL_FOUND)
    set(MFX_INCLUDE_DIRS "")
    set(MFX_LIBRARIES "${VPL_IMPORTED_TARGETS}")
    set(HAVE_MFX TRUE)
    list(APPEND MFX_DEFS "HAVE_ONEVPL")
  endif()
endif()


if(NOT HAVE_MFX)
  set(paths "${MFX_HOME}" ENV "MFX_HOME" ENV "INTELMEDIASDKROOT")
  if(MSVC)
    if(MSVC_VERSION LESS 1900)
      set(vs_suffix)
    else()
      set(vs_suffix "_vs2015")
    endif()
    if(X86_64)
      set(vs_arch "x64")
    else()
      set(vs_arch "win32")
    endif()
  endif()
  find_path(MFX_INCLUDE mfxdefs.h
    PATHS ${paths}
    PATH_SUFFIXES "include" "include/mfx"
    NO_DEFAULT_PATH)
  find_library(MFX_LIBRARY NAMES mfx libmfx${vs_suffix}
    PATHS ${paths}
    PATH_SUFFIXES "lib64" "lib/lin_x64" "lib/${vs_arch}"
    NO_DEFAULT_PATH)
  if(MFX_INCLUDE AND MFX_LIBRARY)
    set(HAVE_MFX TRUE)
    set(MFX_INCLUDE_DIRS "${MFX_INCLUDE}")
    set(MFX_LIBRARIES "${MFX_LIBRARY}")
    list(APPEND MFX_DEFS "HAVE_MFX_PLUGIN")
  endif()
endif()

if(NOT HAVE_MFX AND PKG_CONFIG_FOUND)
  ocv_check_modules(MFX mfx)
endif()

if(HAVE_MFX AND UNIX)
  foreach(mode NO_DEFAULT_PATH "")
    find_path(MFX_va_INCLUDE va/va.h PATHS ${paths} PATH_SUFFIXES "include" ${mode})
    find_library(MFX_va_LIBRARY va PATHS ${paths} PATH_SUFFIXES "lib64" "lib/lin_x64" ${mode})
    find_library(MFX_va_drm_LIBRARY va-drm PATHS ${paths} PATH_SUFFIXES "lib64" "lib/lin_x64" ${mode})
    if(MFX_va_INCLUDE AND MFX_va_LIBRARY AND MFX_va_drm_LIBRARY)
      list(APPEND MFX_INCLUDE_DIRS "${MFX_va_INCLUDE}")
      list(APPEND MFX_LIBRARIES "${MFX_va_LIBRARY}" "${MFX_va_drm_LIBRARY}")
      # list(APPEND MFX_LIBRARIES "-Wl,--exclude-libs=libmfx")
      break()
    endif()
    unset(MFX_va_INCLUDE CACHE)
    unset(MFX_va_LIBRARY CACHE)
    unset(MFX_va_drm_LIBRARY CACHE)
  endforeach()
  if(NOT(MFX_va_INCLUDE AND MFX_va_LIBRARY AND MFX_va_drm_LIBRARY))
    set(HAVE_MFX FALSE)
  endif()

endif()

if(HAVE_MFX)
  list(APPEND MFX_DEFS "HAVE_MFX")
  ocv_add_external_target(mediasdk "${MFX_INCLUDE_DIRS}" "${MFX_LIBRARIES}" "${MFX_DEFS}")
endif()

set(HAVE_MFX ${HAVE_MFX} PARENT_SCOPE)
