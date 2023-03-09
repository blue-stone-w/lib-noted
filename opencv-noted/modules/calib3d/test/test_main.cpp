#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace std;

enum Pattern
{
    CHESSBOARD,
    CIRCLES_GRID,
    ASYMMETRIC_CIRCLES_GRID
};

static bool readStringList(const string &filename, vector<string> &l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    size_t dir_pos = filename.rfind('/');
    if (dir_pos == string::npos)
        dir_pos = filename.rfind('\\');
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
    {
        string fname = (string)*it;
        if (dir_pos != string::npos)
        {
            string fpath = samples::findFile(filename.substr(0, dir_pos + 1) + fname, false);
            if (fpath.empty())
            {
                fpath = samples::findFile(fname);
            }
            fname = fpath;
        }
        else
        {
            fname = samples::findFile(fname);
        }
        l.push_back(fname);
    }
    return true;
}

// 计算重投影误差
static double computeReprojectionErrors(
    const vector<vector<Point3f>> &objectPoints,
    const vector<vector<Point2f>> &imagePoints,
    const vector<Mat> &rvecs, const vector<Mat> &tvecs,
    const Mat &cameraMatrix, const Mat &distCoeffs,
    vector<float> &perViewErrors)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int)objectPoints.size(); i++)
    {
        // 点投影
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f> &corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch (patternType)
    {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                corners.push_back(Point3f(float(j * squareSize),
                                          float(i * squareSize), 0));
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
                                          float(i * squareSize), 0));
        break;

    default:
        CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

static bool runCalibration(vector<vector<Point2f>> imagePoints,
                           Size imageSize, Size boardSize, Pattern patternType,
                           float squareSize, float aspectRatio,
                           float grid_width, bool release_object,
                           int flags, Mat &cameraMatrix, Mat &distCoeffs,
                           vector<Mat> &rvecs, vector<Mat> &tvecs,
                           vector<float> &reprojErrs,
                           vector<Point3f> &newObjPoints,
                           double &totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (flags & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = aspectRatio;

    // distCoeffs = Mat::zeros(8, 1, CV_64F);

    // 计算棋盘格3D坐标
    vector<vector<Point3f>> objectPoints(1);
    // 通过棋盘格的尺寸和行列计算角点坐标
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
    objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    // 相机标定，这个标定函数会使用某种方法更新3D坐标，在标定板不是那么精确的情况下优化标定结果
    // 也可以使用calibrateCamera( ) 替换
    double rms;
    int iFixedPoint = -1;
    if (release_object)
        iFixedPoint = boardSize.width - 1;
    // rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
    //                         cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
    //                         flags | CALIB_FIX_K3 | CALIB_USE_LU);
    //  (3d position, 2d position, , intrin, distort coeff, rotation matrix in Loderiges, translation, control option, end condition)
    rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs,
                          flags | CALIB_FIX_K3 | CALIB_USE_LU);
    // rms = fisheye::calibrate(objectPoints, imagePoints, imageSize,
    //    cameraMatrix, distCoeffs, rvecs, tvecs,
    //    flags | CALIB_FIX_K3 | CALIB_USE_LU);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    if (release_object)
    {
        cout << "New board corners: " << endl;
        cout << newObjPoints[0] << endl;
        cout << newObjPoints[boardSize.width - 1] << endl;
        cout << newObjPoints[boardSize.width * (boardSize.height - 1)] << endl;
        cout << newObjPoints.back() << endl;
    }

    // 计算重投影误差
    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

int main(int argc, char **argv)
{
    // 1. 参数定义
    Size boardSize(9, 6);
    Pattern pattern = CHESSBOARD;        // 标定模板样式，这里是chessboard
    vector<vector<Point2f>> imagePoints; // 图片检测角点
    vector<string> imageList;            // 图片列表
    float squareSize = 20, aspectRatio = 1;
    int winSize = 11;                                      // 窗口大小
    float grid_width = squareSize * (boardSize.width - 1); // grid 宽
    string inputFilename = "../../../samples/data/calibration.yml";

    // 2. 角点检测部分
    readStringList(samples::findFile(inputFilename), imageList);
    // 2.1 循环检测角点
    Size imageSize;
    for (auto &img : imageList)
    {
        Mat view = imread(img, 1); // 读取图片
        imageSize = view.size();
        Mat viewGray;
        cvtColor(view, viewGray, COLOR_BGR2GRAY);
        vector<Point2f> pointbuf;
        // 二值化-->腐蚀和膨胀-->相连棋盘格的断开-->删除非四边形的包络
        bool found = findChessboardCorners(view, boardSize, pointbuf,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        // improve the found corners' coordinate accuracy
        if (pattern == CHESSBOARD && found)
            cornerSubPix(viewGray, pointbuf, Size(winSize, winSize),
                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

        if (found)
            drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

        // 存储角点
        imagePoints.push_back(pointbuf);
    }
    // 4. 标定
    Mat cameraMatrix, distCoeffs; // x参数矩阵
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    vector<Point3f> newObjPoints;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, pattern, squareSize,
                             aspectRatio, grid_width, true, 0, cameraMatrix, distCoeffs,
                             rvecs, tvecs, reprojErrs, newObjPoints, totalAvgErr);
    printf("%s. avg reprojection error = %.7f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    Mat view, rview, map1, map2;
    Mat opt_CameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0); // 保留黑边，获取最大fov去畸变图片
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            opt_CameraMatrix,
                            imageSize, CV_16SC2, map1, map2);
    remap(view, rview, map1, map2, cv::INTER_LINEAR);

    return 0;
}
