
#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/private.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/imgproc/hal/hal.hpp"
#include "hal_replacement.hpp"

#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>

#define GET_OPTIMIZED(func) (func)

/* helper tables */
extern const uchar icvSaturate8u_cv[];
#define CV_FAST_CAST_8U(t) ((-256 <= (t) && (t) <= 512) ? icvSaturate8u_cv[(t) + 256] : 0)
#define CV_CALC_MIN_8U(a, b) (a) -= CV_FAST_CAST_8U((a) - (b))
#define CV_CALC_MAX_8U(a, b) (a) += CV_FAST_CAST_8U((b) - (a))

// -256.f ... 511.f
extern const float icv8x32fTab_cv[];
#define CV_8TO32F(x) icv8x32fTab_cv[(x) + 256]

// (-128.f)^2 ... (255.f)^2
extern const float icv8x32fSqrTab[];
#define CV_8TO32F_SQR(x) icv8x32fSqrTab[(x) + 128]

#define CV_COPY(dst, src, len, idx)         \
    for ((idx) = 0; (idx) < (len); (idx)++) \
    (dst)[idx] = (src)[idx]

#define CV_SET(dst, val, len, idx)          \
    for ((idx) = 0; (idx) < (len); (idx)++) \
    (dst)[idx] = (val)

#undef CV_CALC_MIN
#define CV_CALC_MIN(a, b) \
    if ((a) > (b))        \
    (a) = (b)

#undef CV_CALC_MAX
#define CV_CALC_MAX(a, b) \
    if ((a) < (b))        \
    (a) = (b)

#ifdef HAVE_IPP
static inline IppiInterpolationType ippiGetInterpolation(int inter)
{
    inter &= cv::INTER_MAX;
    return inter == cv::INTER_NEAREST ? ippNearest : inter == cv::INTER_LINEAR ? ippLinear
                                                 : inter == cv::INTER_CUBIC    ? ippCubic
                                                 : inter == cv::INTER_LANCZOS4 ? ippLanczos
                                                 : inter == cv::INTER_AREA     ? ippSuper
                                                                               : (IppiInterpolationType)-1;
}
#endif

#include "_geom.h"
#include "filterengine.hpp"

#include "opencv2/core/sse_utils.hpp"

inline bool isStorageOrMat(void *arr)
{
    if (CV_IS_STORAGE(arr))
        return true;
    else if (CV_IS_MAT(arr))
        return false;
    CV_Error(CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*");
}

#endif /*__OPENCV_CV_INTERNAL_H_*/
