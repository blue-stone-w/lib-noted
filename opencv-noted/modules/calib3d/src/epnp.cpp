#include <iostream>
#include "precomp.hpp"
#include "epnp.h"

namespace cv
{

  epnp::epnp(const Mat &cameraMatrix, const Mat &opoints, const Mat &ipoints)
  {
    if (cameraMatrix.depth() == CV_32F)
    {
      init_camera_parameters<float>(cameraMatrix);
    }
    else
    {
      init_camera_parameters<double>(cameraMatrix);
    }
    // checkVector: (elemChannels, depth); -1 if the requirement is not satisfied.
    // Otherwise, returns the number of elements in the matrix. Note that an element may have multiple channels.
    // Here an element is a point
    number_of_correspondences = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));

    pws.resize(3 * number_of_correspondences);
    us.resize(2 * number_of_correspondences);

    // extract data according to mat type
    if (opoints.depth() == ipoints.depth())
    {
      if (opoints.depth() == CV_32F)
      {
        init_points<Point3f, Point2f>(opoints, ipoints);
      }
      else
      {
        init_points<Point3d, Point2d>(opoints, ipoints);
      }
    }
    else if (opoints.depth() == CV_32F)
    {
      init_points<Point3f, Point2d>(opoints, ipoints);
    }
    else
    {
      init_points<Point3d, Point2f>(opoints, ipoints);
    }

    alphas.resize(4 * number_of_correspondences);
    pcs.resize(3 * number_of_correspondences);

    max_nr = 0;
    A1 = NULL;
    A2 = NULL;
  }

  epnp::~epnp()
  {
    if (A1)
    {
      delete[] A1;
    }
    if (A2)
    {
      delete[] A2;
    }
  }

  void epnp::choose_control_points(void)
  {
    // Take C0 as the reference points centroid:
    cws[0][0] = cws[0][1] = cws[0][2] = 0;
    for (int i = 0; i < number_of_correspondences; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        cws[0][j] += pws[3 * i + j]; // sum of obj pts
      }
    }

    for (int j = 0; j < 3; j++)
    {
      cws[0][j] /= number_of_correspondences; // average
    }

    // Take C1, C2, and C3 from PCA on the reference points:
    CvMat *PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F); // (row, col, type); relative coordinate

    double pw0tpw0[3 * 3] = {}; //
    double dc[3] = {};          // eigen value
    double uct[3 * 3] = {};     // eigen vector

    CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0); // (row, col, type, data): create mat from array
    CvMat DC = cvMat(3, 1, CV_64F, dc);
    CvMat UCt = cvMat(3, 3, CV_64F, uct); // array and mat share memory

    // coordinate relative to reference points(centroid)
    for (int i = 0; i < number_of_correspondences; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];
      }
    }

    // 计算输入矩阵和它的转置的乘积:(输入矩阵,目标矩阵,乘法顺序,一个可选数组(在乘法之前从 输入矩阵 中减去该数组)。
    cvMulTransposed(PW0, &PW0tPW0, 1);
    // (输入矩阵,结果奇异值矩阵,可选的左部正交矩阵,可选右部正交矩阵,flag): 奇异值都是非负的并按降序存储
    // CV_SVD_MODIFY_A 通过操作可以修改矩阵 src1; CV_SVD_U_T 意味着会返回转置矩阵 U; CV_SVD_V_T 意味着会返回转置矩阵 V;
    cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

    cvReleaseMat(&PW0); // delete matrix pointer

    // calculate control points
    for (int i = 1; i < 4; i++)
    {
      double k = sqrt(dc[i - 1] / number_of_correspondences); // sqrt(singular values)
      for (int j = 0; j < 3; j++)
      {
        cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
      }
    }
  }

  void epnp::compute_barycentric_coordinates(void)
  {
    double cc[3 * 3] = {}; // a set of basis by contronl points
    double cc_inv[3 * 3] = {};
    CvMat CC = cvMat(3, 3, CV_64F, cc);
    CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

    // calculate this set of basis
    for (int i = 0; i < 3; i++)
    {
      for (int j = 1; j < 4; j++)
      {
        cc[3 * i + j - 1] = cws[j][i] - cws[0][i]; // control - centroid
      }
    }

    cvInvert(&CC, &CC_inv, CV_SVD); // (src, dst, method): get invert of src
    double *ci = cc_inv;
    // calculate new coordinates of 3d points in this new set of basis
    for (int i = 0; i < number_of_correspondences; i++)
    {
      double *pi = &pws[0] + 3 * i;   // pw s: all obj pts
      double *a = &alphas[0] + 4 * i; // assign using pointer

      for (int j = 0; j < 3; j++)
      {
        a[1 + j] = ci[3 * j + 0] * (pi[0] - cws[0][0]) +
                   ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                   ci[3 * j + 2] * (pi[2] - cws[0][2]); //
      }
      a[0] = 1.0f - a[1] - a[2] - a[3];
    }
  }

  // (out M, in row, in new coordinate, in u, in v): fill 2 rows for a point according to equation 567
  void epnp::fill_M(CvMat *M, const int row, const double *as, const double u, const double v)
  {
    // db is first position of data; 12 is num of column; 12 = 3(xyz) * 4(num of control points)
    double *M1 = M->data.db + row * 12; // u(3), first row
    double *M2 = M1 + 12;               // v(3), second row

    // transform to camera coordinate system with intrinsic
    for (int i = 0; i < 4; i++) // 4 control points
    {
      // equation 5
      M1[3 * i] = as[i] * fu;
      M1[3 * i + 1] = 0.0;
      M1[3 * i + 2] = as[i] * (uc - u);
      // equation 6
      M2[3 * i] = 0.0;
      M2[3 * i + 1] = as[i] * fv;
      M2[3 * i + 2] = as[i] * (vc - v);
    }
  }

  void epnp::compute_ccs(const double *betas, const double *ut)
  {
    // initialize control points in camera
    for (int i = 0; i < 4; i++)
    {
      ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;
    }

    // 4 betas
    for (int i = 0; i < 4; i++)
    {
      const double *v = ut + 12 * (11 - i);
      // 4 control points
      for (int j = 0; j < 4; j++)
      {
        // 3: xyz
        for (int k = 0; k < 3; k++)
        {
          ccs[j][k] += betas[i] * v[3 * j + k];
        }
      }
    }
  }

  void epnp::compute_pcs(void)
  {
    for (int i = 0; i < number_of_correspondences; i++)
    {
      double *a = &alphas[0] + 4 * i;
      double *pc = &pcs[0] + 3 * i;

      for (int j = 0; j < 3; j++)
      {
        pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
      }
    }
  }

  void epnp::compute_pose(Mat &R, Mat &t)
  {
    choose_control_points();
    // get coordinates of pw in the set of basis from control points. I called them new coordinates
    compute_barycentric_coordinates();

    // project control points to camera
    CvMat *M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);
    for (int i = 0; i < number_of_correspondences; i++)
    {
      fill_M(M, 2 * i, &alphas[0] + 4 * i, us[2 * i], us[2 * i + 1]);
    }

    double mtm[12 * 12] = {};
    double d[12] = {};
    double ut[12 * 12] = {};
    CvMat MtM = cvMat(12, 12, CV_64F, mtm);
    CvMat D = cvMat(12, 1, CV_64F, d);
    CvMat Ut = cvMat(12, 12, CV_64F, ut); // vi in article

    // 计算输入矩阵和它的转置的乘积:(输入矩阵,目标矩阵,乘法顺序,一个可选数组); 一个可选数组:在乘法之前从 输入矩阵 中减去该数组
    cvMulTransposed(M, &MtM, 1);
    // (输入矩阵,结果奇异值矩阵,可选的左部正交矩阵,可选右部正交矩阵,flag): 奇异值都是非负的并按降序存储
    cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
    cvReleaseMat(&M); // delete matrix pointer

    // l&r: we can get 6 distance equations from 4 control points, every row is a distance equation
    // expand eq 12(but use N=4), we can get 10 combination, which are multiplication of two control points
    double l_6x10[6 * 10] = {};
    // distances between control points in world frame; 4 control points -> 6 combination
    double rho[6] = {};
    CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
    CvMat Rho = cvMat(6, 1, CV_64F, rho);

    // Calculate l in case N = 4. l: left term of equation is square of distance in camera frame.
    compute_L_6x10(ut, l_6x10);
    // calculate distances between control points in world frame
    compute_rho(rho);

    double Betas[4][4] = {}; // 1st row <-> N = 1; 2nd row <-> N = 2; 3rd<-> N = 3
    double rep_errors[4] = {};
    double Rs[4][3][3] = {}, ts[4][3] = {};

    // approximate value
    find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
    // accurate value
    gauss_newton(&L_6x10, &Rho, Betas[1]);
    // errors
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

    find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
    gauss_newton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

    find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
    gauss_newton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

    // select N that yields the smallest reprojection error
    int N = 1;
    if (rep_errors[2] < rep_errors[1])
    {
      N = 2;
    }
    if (rep_errors[3] < rep_errors[N])
    {
      N = 3;
    }

    Mat(3, 1, CV_64F, ts[N]).copyTo(t);
    Mat(3, 3, CV_64F, Rs[N]).copyTo(R);
  }

  void epnp::copy_R_and_t(const double R_src[3][3], const double t_src[3],
                          double R_dst[3][3], double t_dst[3])
  {
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
        R_dst[i][j] = R_src[i][j];
      t_dst[i] = t_src[i];
    }
  }

  double epnp::dist2(const double *p1, const double *p2)
  {
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) +
           (p1[1] - p2[1]) * (p1[1] - p2[1]) +
           (p1[2] - p2[2]) * (p1[2] - p2[2]);
  }

  double epnp::dot(const double *v1, const double *v2)
  {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  }

  void epnp::estimate_R_and_t(double R[3][3], double t[3])
  {
    double pc0[3] = {}, pw0[3] = {};
    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;

    for (int i = 0; i < number_of_correspondences; i++)
    {
      const double *pc = &pcs[3 * i];
      const double *pw = &pws[3 * i];

      // 3: xyz
      for (int j = 0; j < 3; j++)
      {
        pc0[j] += pc[j];
        pw0[j] += pw[j];
      }
    }
    // center
    for (int j = 0; j < 3; j++)
    {
      pc0[j] /= number_of_correspondences;
      pw0[j] /= number_of_correspondences;
    }

    double abt[3 * 3] = {}, abt_d[3] = {}, abt_u[3 * 3] = {}, abt_v[3 * 3] = {};
    CvMat ABt = cvMat(3, 3, CV_64F, abt);
    CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
    CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
    CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

    cvSetZero(&ABt);
    for (int i = 0; i < number_of_correspondences; i++)
    {
      double *pc = &pcs[3 * i];
      double *pw = &pws[3 * i];

      for (int j = 0; j < 3; j++)
      {
        abt[3 * j + 0] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
        abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
        abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
      }
    }
    // (输入矩阵,结果奇异值矩阵,可选的左部正交矩阵,可选右部正交矩阵,flag): 奇异值都是非负的并按降序存储
    cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);
      }
    }

    const double det = R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
                       R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

    if (det < 0)
    {
      R[2][0] = -R[2][0];
      R[2][1] = -R[2][1];
      R[2][2] = -R[2][2];
    }

    t[0] = pc0[0] - dot(R[0], pw0);
    t[1] = pc0[1] - dot(R[1], pw0);
    t[2] = pc0[2] - dot(R[2], pw0);
  }

  void epnp::solve_for_sign(void)
  {
    if (pcs[2] < 0.0)
    {
      for (int i = 0; i < 4; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          ccs[i][j] = -ccs[i][j];
        }
      }

      for (int i = 0; i < number_of_correspondences; i++)
      {
        pcs[3 * i] = -pcs[3 * i];
        pcs[3 * i + 1] = -pcs[3 * i + 1];
        pcs[3 * i + 2] = -pcs[3 * i + 2];
      }
    }
  }

  double epnp::compute_R_and_t(const double *ut, const double *betas, double R[3][3], double t[3])
  {
    compute_ccs(betas, ut);
    compute_pcs();

    solve_for_sign();

    estimate_R_and_t(R, t);

    return reprojection_error(R, t);
  }

  double epnp::reprojection_error(const double R[3][3], const double t[3])
  {
    double sum2 = 0.0; // standard deviation

    for (int i = 0; i < number_of_correspondences; i++)
    {
      double *pw = &pws[3 * i];
      double Xc = dot(R[0], pw) + t[0]; // transfrom from world frame to camera frame
      double Yc = dot(R[1], pw) + t[1];
      double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);

      double ue = uc + fu * Xc * inv_Zc; // projection
      double ve = vc + fv * Yc * inv_Zc;
      double u = us[2 * i], v = us[2 * i + 1]; // original pixel coordinates

      sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
    }

    return sum2 / number_of_correspondences;
  }

  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_1 = [B11 B12     B13         B14]
  void epnp::find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho, double *betas)
  {
    double l_6x4[6 * 4] = {}, b4[4] = {};
    CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
    CvMat B4 = cvMat(4, 1, CV_64F, b4);

    // parse value: distaces of 6 control point pairs from 4 points
    for (int i = 0; i < 6; i++)
    {
      // set(mat, row, col, value); value = get(mat, row, col)
      cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0)); // 00
      cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1)); // 01
      cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3)); // 02
      cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6)); // 03
    }

    cvSolve(&L_6x4, Rho, &B4, CV_SVD);

    if (b4[0] < 0)
    {
      betas[0] = sqrt(-b4[0]);
      betas[1] = -b4[1] / betas[0];
      betas[2] = -b4[2] / betas[0];
      betas[3] = -b4[3] / betas[0];
    }
    else
    {
      betas[0] = sqrt(b4[0]);
      betas[1] = b4[1] / betas[0];
      betas[2] = b4[2] / betas[0];
      betas[3] = b4[3] / betas[0];
    }
  }

  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_2 = [B11 B12 B22                            ]
  void epnp::find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho, double *betas)
  {
    double l_6x3[6 * 3] = {}, b3[3] = {};
    CvMat L_6x3 = cvMat(6, 3, CV_64F, l_6x3);
    CvMat B3 = cvMat(3, 1, CV_64F, b3);

    for (int i = 0; i < 6; i++)
    {
      cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
      cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
      cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
    }

    cvSolve(&L_6x3, Rho, &B3, CV_SVD);

    if (b3[0] < 0)
    {
      betas[0] = sqrt(-b3[0]);
      betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    }
    else
    {
      betas[0] = sqrt(b3[0]);
      betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0)
    {
      betas[0] = -betas[0];
    }

    betas[2] = 0.0;
    betas[3] = 0.0;
  }

  // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
  // betas_approx_3 = [B11 B12 B22 B13 B23                    ]
  void epnp::find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho,
                                 double *betas)
  {
    double l_6x5[6 * 5] = {}, b5[5] = {};
    CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
    CvMat B5 = cvMat(5, 1, CV_64F, b5);

    for (int i = 0; i < 6; i++)
    {
      cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
      cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
      cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
      cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
      cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
    }

    cvSolve(&L_6x5, Rho, &B5, CV_SVD);

    if (b5[0] < 0)
    {
      betas[0] = sqrt(-b5[0]);
      betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    }
    else
    {
      betas[0] = sqrt(b5[0]);
      betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0)
    {
      betas[0] = -betas[0];
    }
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
  }

  void epnp::compute_L_6x10(const double *ut, double *l_6x10)
  {
    const double *v[4];

    v[0] = ut + 12 * 11; // v[0] is an array
    v[1] = ut + 12 * 10; // also an array
    v[2] = ut + 12 * 9;
    v[3] = ut + 12 * 8;

    double dv[4][6][3] = {};

    for (int i = 0; i < 4; i++) // 4: 4 control points
    {
      int a = 0, b = 1;
      for (int j = 0; j < 6; j++) // 6: 6 combination
      {
        dv[i][j][0] = v[i][3 * a + 0] - v[i][3 * b + 0]; // d012: xyz
        dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
        dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

        b++;
        if (b > 3)
        {
          a++;
          b = a + 1;
        }
        // ab: 01 02 03 12 13 23
      }
    } // endfor:

    // 6 combination <-> 6 equations
    for (int i = 0; i < 6; i++)
    {
      double *row = l_6x10 + 10 * i;

      row[0] = dot(dv[0][i], dv[0][i]);        // 00
      row[1] = 2.0f * dot(dv[0][i], dv[1][i]); // 01
      row[2] = dot(dv[1][i], dv[1][i]);        // 11
      row[3] = 2.0f * dot(dv[0][i], dv[2][i]); // 02
      row[4] = 2.0f * dot(dv[1][i], dv[2][i]); // 12
      row[5] = dot(dv[2][i], dv[2][i]);        // 22
      row[6] = 2.0f * dot(dv[0][i], dv[3][i]); // 03
      row[7] = 2.0f * dot(dv[1][i], dv[3][i]); // 13
      row[8] = 2.0f * dot(dv[2][i], dv[3][i]); // 23
      row[9] = dot(dv[3][i], dv[3][i]);        // 33

    } // endfor:
  }

  // distances between control points in world frame
  void epnp::compute_rho(double *rho)
  {
    rho[0] = dist2(cws[0], cws[1]); // square of distance of control points in world frame
    rho[1] = dist2(cws[0], cws[2]); // 2-norm
    rho[2] = dist2(cws[0], cws[3]); // we can get 6 distances from 4 control points
    rho[3] = dist2(cws[1], cws[2]);
    rho[4] = dist2(cws[1], cws[3]);
    rho[5] = dist2(cws[2], cws[3]);
  }

  void epnp::compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                          const double betas[4], CvMat *A, CvMat *b)
  {
    for (int i = 0; i < 6; i++) // 6 equations
    {
      const double *rowL = l_6x10 + i * 10; // reference a row from l
      double *rowA = A->data.db + i * 4;

      // Jacobian of eq12's left, with respect to beta1, beta2, beta3, beta4
      rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
      rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
      rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
      rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

      // residual
      cvmSet(b, i, 0,
             rho[i] - (rowL[0] * betas[0] * betas[0] +
                       rowL[1] * betas[0] * betas[1] +
                       rowL[2] * betas[1] * betas[1] +
                       rowL[3] * betas[0] * betas[2] +
                       rowL[4] * betas[1] * betas[2] +
                       rowL[5] * betas[2] * betas[2] +
                       rowL[6] * betas[0] * betas[3] +
                       rowL[7] * betas[1] * betas[3] +
                       rowL[8] * betas[2] * betas[3] +
                       rowL[9] * betas[3] * betas[3]));
    }
  }

  void epnp::gauss_newton(const CvMat *L_6x10, const CvMat *Rho, double betas[4])
  {
    const int iterations_number = 5;

    double a[6 * 4] = {}, b[6] = {}, x[4] = {};
    CvMat A = cvMat(6, 4, CV_64F, a); // Jacobian
    CvMat B = cvMat(6, 1, CV_64F, b); // residual
    CvMat X = cvMat(4, 1, CV_64F, x); // delta_x

    for (int k = 0; k < iterations_number; k++)
    {
      compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, betas, &A, &B);
      qr_solve(&A, &B, &X);
      for (int i = 0; i < 4; i++)
      {
        betas[i] += x[i];
      }
    }
  }

  // A:6*4; B:6*1; X:4*1; AX=B
  void epnp::qr_solve(CvMat *A, CvMat *b, CvMat *X)
  {
    const int nr = A->rows; // ???
    const int nc = A->cols;
    if (nc <= 0 || nr <= 0)
    {
      return;
    }

    if (max_nr != 0 && max_nr < nr)
    {
      delete[] A1;
      delete[] A2;
    }
    if (max_nr < nr)
    {
      max_nr = nr;
      A1 = new double[nr];
      A2 = new double[nr];
    }

    double *pA = A->data.db;
    double *ppAkk = pA; // assignment, don't share memory

    for (int k = 0; k < nc; k++)
    {
      double *ppAik1 = ppAkk;
      double eta = fabs(*ppAik1);
      // traverse every row in this column
      for (int i = k + 1; i < nr; i++)
      {
        double elt = fabs(*ppAik1);
        // get max in this column
        if (eta < elt)
        {
          eta = elt;
        }
        ppAik1 += nc; // point to next row in the same column
      }
      if (eta == 0)
      {
        A1[k] = A2[k] = 0.0; //???
        // cerr << "God damnit, A is singular, this shouldn't happen." << endl;
        return;
      }
      else
      {
        double *ppAik2 = ppAkk;
        double sum2 = 0.0;
        double inv_eta = 1. / eta;
        // k is index of column
        for (int i = k; i < nr; i++)
        {
          *ppAik2 *= inv_eta;
          sum2 += *ppAik2 * *ppAik2;
          ppAik2 += nc; // point to next row in the same column
        }
        double sigma = sqrt(sum2);
        if (*ppAkk < 0)
        {
          sigma = -sigma;
        }
        *ppAkk += sigma;
        A1[k] = sigma * *ppAkk;
        A2[k] = -eta * sigma;
        for (int j = k + 1; j < nc; j++)
        {
          double *ppAik = ppAkk, sum = 0;
          for (int i = k; i < nr; i++)
          {
            sum += *ppAik * ppAik[j - k];
            ppAik += nc;
          }
          double tau = sum / A1[k];
          ppAik = ppAkk;
          for (int i = k; i < nr; i++)
          {
            ppAik[j - k] -= tau * *ppAik;
            ppAik += nc;
          }
        }
      } // end else:
      ppAkk += nc + 1;
    } // endfor: have traversed every column

    // b <- Qt b
    double *ppAjj = pA, *pb = b->data.db;
    for (int j = 0; j < nc; j++)
    {
      double *ppAij = ppAjj, tau = 0;
      for (int i = j; i < nr; i++)
      {
        tau += *ppAij * pb[i];
        ppAij += nc;
      }
      tau /= A1[j];
      ppAij = ppAjj;
      for (int i = j; i < nr; i++)
      {
        pb[i] -= tau * *ppAij;
        ppAij += nc;
      }
      ppAjj += nc + 1;
    }

    // X = R-1 b
    double *pX = X->data.db;
    pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
    for (int i = nc - 2; i >= 0; i--)
    {
      double *ppAij = pA + i * nc + (i + 1), sum = 0;

      for (int j = i + 1; j < nc; j++)
      {
        sum += *ppAij * pX[j];
        ppAij++;
      }
      pX[i] = (pb[i] - sum) / A2[i];
    }
  }

}
