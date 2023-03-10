// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_IEWRAPPER_HPP
#define OPENCV_GAPI_IEWRAPPER_HPP

#ifdef HAVE_INF_ENGINE

#include <inference_engine.hpp>

#include <vector>
#include <string>
#include <fstream>

#include "opencv2/gapi/infer/ie.hpp"

namespace IE = InferenceEngine;
using GIEParam = cv::gapi::ie::detail::ParamDesc;

namespace cv {
namespace gimpl {
namespace ie {
namespace wrap {
// NB: These functions are EXPORTed to make them accessible by the
// test suite only.
GAPI_EXPORTS std::vector<std::string> getExtensions(const GIEParam& params);
GAPI_EXPORTS IE::CNNNetwork readNetwork(const GIEParam& params);

IE::InputsDataMap  toInputsDataMap (const IE::ConstInputsDataMap& inputs);
IE::OutputsDataMap toOutputsDataMap(const IE::ConstOutputsDataMap& outputs);

#if INF_ENGINE_RELEASE < 2019020000  // < 2019.R2
using Plugin = IE::InferencePlugin;
GAPI_EXPORTS IE::InferencePlugin getPlugin(const GIEParam& params);
GAPI_EXPORTS inline IE::ExecutableNetwork loadNetwork(      IE::InferencePlugin& plugin,
                                                      const IE::CNNNetwork&      net,
                                                      const GIEParam&) {
    return plugin.LoadNetwork(net, {}); // FIXME: 2nd parameter to be
                                        // configurable via the API
}
GAPI_EXPORTS inline IE::ExecutableNetwork importNetwork(      IE::CNNNetwork& plugin,
                                                        const GIEParam& param) {
    return plugin.ImportNetwork(param.model_path, param.device_id, {});
}
#else // >= 2019.R2
using Plugin = IE::Core;
GAPI_EXPORTS IE::Core getCore();
GAPI_EXPORTS IE::Core getPlugin(const GIEParam& params);
GAPI_EXPORTS inline IE::ExecutableNetwork loadNetwork(      IE::Core&       core,
                                                      const IE::CNNNetwork& net,
                                                      const GIEParam& params,
                                                      IE::RemoteContext::Ptr rctx = nullptr) {
    if (rctx != nullptr) {
        return core.LoadNetwork(net, rctx);
    } else {
        return core.LoadNetwork(net, params.device_id);
    }
}
GAPI_EXPORTS inline IE::ExecutableNetwork importNetwork(      IE::Core& core,
                                                        const GIEParam& params,
                                                        IE::RemoteContext::Ptr rctx = nullptr) {
    if (rctx != nullptr) {
        std::filebuf blobFile;
        if (!blobFile.open(params.model_path, std::ios::in | std::ios::binary))
        {
            blobFile.close();
            throw std::runtime_error("Could not open file");
        }
        std::istream graphBlob(&blobFile);
        return core.ImportNetwork(graphBlob, rctx);
    } else {
        return core.ImportNetwork(params.model_path, params.device_id, {});
    }
}
#endif // INF_ENGINE_RELEASE < 2019020000
}}}}

#endif //HAVE_INF_ENGINE
#endif // OPENCV_GAPI_IEWRAPPER_HPP
