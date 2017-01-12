/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_FEATURES_IMPL_PRINCIPAL_CURVATURES_H_
#define PCL_FEATURES_IMPL_PRINCIPAL_CURVATURES_H_

#include <pcl/features/principal_curvatures.h>

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT> void
pcl::PrincipalCurvaturesEstimation<PointInT, PointNT, PointOutT>::computePointPrincipalCurvatures (
      const pcl::PointCloud<PointNT> &normals, int p_idx, const std::vector<int> &indices,
      float &pcx, float &pcy, float &pcz, float &pc1, float &pc2)
{
  EIGEN_ALIGN16 Eigen::Matrix3f I = Eigen::Matrix3f::Identity ();
  Eigen::Vector3f n_idx (normals.points[p_idx].normal[0], normals.points[p_idx].normal[1], normals.points[p_idx].normal[2]);
  EIGEN_ALIGN16 Eigen::Matrix3f M = I - n_idx * n_idx.transpose ();    // projection matrix (into tangent plane)

  // Project normals into the tangent plane
  Eigen::Vector3f normal;
  projected_normals_.resize (indices.size ());
  xyz_centroid_.setZero ();
  for (size_t idx = 0; idx < indices.size(); ++idx)
  {
    normal[0] = normals.points[indices[idx]].normal[0];
    normal[1] = normals.points[indices[idx]].normal[1];
    normal[2] = normals.points[indices[idx]].normal[2];

    projected_normals_[idx] = M * normal;
    xyz_centroid_ += projected_normals_[idx];
  }

  // Estimate the XYZ centroid
  xyz_centroid_ /= static_cast<float> (indices.size ());

  // Initialize to 0
  covariance_matrix_.setZero ();

  double demean_xy, demean_xz, demean_yz;
  // For each point in the cloud
  for (size_t idx = 0; idx < indices.size (); ++idx)
  {
    demean_ = projected_normals_[idx] - xyz_centroid_;

    demean_xy = demean_[0] * demean_[1];
    demean_xz = demean_[0] * demean_[2];
    demean_yz = demean_[1] * demean_[2];

    covariance_matrix_(0, 0) += demean_[0] * demean_[0];
    covariance_matrix_(0, 1) += static_cast<float> (demean_xy);
    covariance_matrix_(0, 2) += static_cast<float> (demean_xz);

    covariance_matrix_(1, 0) += static_cast<float> (demean_xy);
    covariance_matrix_(1, 1) += demean_[1] * demean_[1];
    covariance_matrix_(1, 2) += static_cast<float> (demean_yz);

    covariance_matrix_(2, 0) += static_cast<float> (demean_xz);
    covariance_matrix_(2, 1) += static_cast<float> (demean_yz);
    covariance_matrix_(2, 2) += demean_[2] * demean_[2];
  }

  // Extract the eigenvalues and eigenvectors
  pcl::eigen33 (covariance_matrix_, eigenvalues_);
  pcl::computeCorrespondingEigenVector (covariance_matrix_, eigenvalues_ [2], eigenvector_);

  pcx = eigenvector_ [0];
  pcy = eigenvector_ [1];
  pcz = eigenvector_ [2];
  float indices_size = 1.0f / static_cast<float> (indices.size ());
  pc1 = eigenvalues_ [2] * indices_size;
  pc2 = eigenvalues_ [1] * indices_size;
}


//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT> void
pcl::PrincipalCurvaturesEstimation<PointInT, PointNT, PointOutT>::computeFeature (PointCloudOut &output)
{
  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  std::vector<int> nn_indices (k_);
  std::vector<float> nn_dists (k_);

  output.is_dense = true;
  // Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
  if (input_->is_dense)
  {
    // Iterating over the entire index vector
    for (size_t idx = 0; idx < indices_->size (); ++idx)
    {
      if (this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0)
      {
        output.points[idx].principal_curvature[0] = output.points[idx].principal_curvature[1] = output.points[idx].principal_curvature[2] =
          output.points[idx].pc1 = output.points[idx].pc2 = std::numeric_limits<float>::quiet_NaN ();
        output.is_dense = false;
        continue;
      }

      // Estimate the principal curvatures at each patch
      computePointPrincipalCurvatures (*normals_, (*indices_)[idx], nn_indices,
                                       output.points[idx].principal_curvature[0], output.points[idx].principal_curvature[1], output.points[idx].principal_curvature[2],
                                       output.points[idx].pc1, output.points[idx].pc2);
    }
  }
  else
  {
    // Iterating over the entire index vector
    for (size_t idx = 0; idx < indices_->size (); ++idx)
    {
      if (!isFinite ((*input_)[(*indices_)[idx]]) ||
          this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0)
      {
        output.points[idx].principal_curvature[0] = output.points[idx].principal_curvature[1] = output.points[idx].principal_curvature[2] =
          output.points[idx].pc1 = output.points[idx].pc2 = std::numeric_limits<float>::quiet_NaN ();
        output.is_dense = false;
        continue;
      }

      // Estimate the principal curvatures at each patch
      computePointPrincipalCurvatures (*normals_, (*indices_)[idx], nn_indices,
                                       output.points[idx].principal_curvature[0], output.points[idx].principal_curvature[1], output.points[idx].principal_curvature[2],
                                       output.points[idx].pc1, output.points[idx].pc2);
    }
  }
}





//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT> void
pcl::PrincipalCurvaturesEstimationMLS<PointInT, PointNT, PointOutT>::computePointPrincipalCurvatures (
    const pcl::PointCloud<PointNT> &normals,
    int p_idx, const std::vector<int> &indices,
    float &pc1, float &pc2)
{
  const Eigen::Vector3d eval_pnt (input_->at (p_idx).x, input_->at (p_idx).y, input_->at (p_idx).z);
  const size_t number_of_neighbors = indices.size ();

  Eigen::Vector3d eval_normal (0, 0, 0);

  Eigen::MatrixXd neighbor_normals (3, number_of_neighbors);
  Eigen::MatrixXd diff_vec (3, number_of_neighbors);
  Eigen::RowVectorXd weight (number_of_neighbors);

  for (size_t i = 0; i < number_of_neighbors; ++i)
  {
    const int ind = indices.at (i);
    const Eigen::Vector3d pt_neighbor (input_->at (ind).x, input_->at (ind).y, input_->at (ind).z);
    diff_vec.col (i) = eval_pnt - pt_neighbor;
    const double dist_sq_i = diff_vec.col (i).squaredNorm ();
    weight (i) = std::exp (-dist_sq_i / gaussian_param_sq_);
    neighbor_normals.col (i) = Eigen::Vector3d (normals[ind].normal_x, normals[ind].normal_y, normals[ind].normal_z);

    eval_normal += weight (i) * neighbor_normals.col (i);
  }

  const Eigen::Vector3d normalized_eval_normal = eval_normal.normalized ();

  const Eigen::RowVectorXd projected_diff_vec =
      diff_vec.row (0) * normalized_eval_normal (0) +
      diff_vec.row (1) * normalized_eval_normal (1) +
      diff_vec.row (2) * normalized_eval_normal (2);

  Eigen::Vector3d delta_g (0, 0, 0);
  Eigen::RowVectorXd term_1st (number_of_neighbors);
  term_1st = projected_diff_vec.array ().square () / gaussian_param_sq_ - 1;
  term_1st = 2 * weight.cwiseProduct (2 / gaussian_param_sq_ * projected_diff_vec.cwiseProduct (term_1st));

  for (int ii = 0; ii < 3; ++ii)
  {
    delta_g (ii) = term_1st.cwiseProduct (diff_vec.row (ii)).sum ();
  }

  Eigen::RowVectorXd term_2nd (number_of_neighbors);
  term_2nd = 1 - 3 / gaussian_param_sq_ * projected_diff_vec.array ().square ();
  term_2nd = 2 * weight.cwiseProduct (term_2nd);

  Eigen::Matrix3d temp;
  for (int ii = 0; ii < 3; ++ii)
    for (int jj = 0; jj < 3; ++jj)
      temp (ii, jj) = (-2 / gaussian_param_sq_ * weight.cwiseProduct (neighbor_normals.row (ii).cwiseProduct (diff_vec.row (jj)))).sum ();
  Eigen::MatrixXd temp_2 (3, number_of_neighbors);
  temp_2 = (Eigen::Matrix3d::Identity () - normalized_eval_normal * normalized_eval_normal.transpose ()) * temp.transpose () / eval_normal.norm () * diff_vec;
  for (int ii = 0; ii < 3; ++ii)
    delta_g (ii) += term_2nd.sum () * normalized_eval_normal (ii) + term_2nd.cwiseProduct (temp_2.row (ii)).sum ();

  Eigen::Matrix3d delta2_g;
  delta2_g.setZero ();

  const Eigen::Matrix3d delta_eval_normal = (Eigen::Matrix3d::Identity () - normalized_eval_normal * normalized_eval_normal.transpose ()) * temp
      / eval_normal.norm ();

  term_1st = projected_diff_vec.array ().square () / gaussian_param_sq_ - 1;
  term_1st = -4 / gaussian_param_sq_ * weight.cwiseProduct (2 / gaussian_param_sq_ * projected_diff_vec.cwiseProduct (term_1st));

  for (int ii = 0; ii < number_of_neighbors; ++ii)
    delta2_g += term_1st (ii) * (diff_vec.col (ii) * diff_vec.col (ii).transpose ());

  term_2nd = 1 - 3 * projected_diff_vec.array ().square () / gaussian_param_sq_;
  term_2nd = -4 / gaussian_param_sq_ * weight.cwiseProduct (term_2nd);

  for (int ii = 0; ii < number_of_neighbors; ++ii)
    delta2_g += term_2nd (ii) * (normalized_eval_normal + delta_eval_normal.transpose () * diff_vec.col (ii)) * diff_vec.col (ii).transpose ();

  Eigen::RowVectorXd term_3rd (number_of_neighbors);
  term_3rd = 6 * projected_diff_vec.array ().square () / (gaussian_param_sq_ * gaussian_param_sq_) - 2 / gaussian_param_sq_;
  term_3rd = 2 * weight.cwiseProduct (term_3rd);

  for (int ii = 0; ii < number_of_neighbors; ++ii)
    delta2_g += term_3rd (ii) * diff_vec.col (ii) * (diff_vec.col (ii).transpose () * delta_eval_normal + normalized_eval_normal.transpose ());

  Eigen::RowVectorXd term_4th (number_of_neighbors);
  term_4th = projected_diff_vec.array ().square () / gaussian_param_sq_ - 1;
  term_4th = 4 / gaussian_param_sq_ * weight.cwiseProduct (projected_diff_vec).cwiseProduct (term_4th);

  for (int ii = 0; ii < number_of_neighbors; ++ii)
    delta2_g += term_4th (ii) * Eigen::Matrix3d::Identity ();

  Eigen::RowVectorXd term_5th (number_of_neighbors);
  term_5th = -12 / gaussian_param_sq_ * weight.cwiseProduct (projected_diff_vec);

  for (int ii = 0; ii < number_of_neighbors; ++ii)
    delta2_g += term_5th (ii) * (normalized_eval_normal + delta_eval_normal.transpose () * diff_vec.col (ii))
        * (diff_vec.col (ii).transpose () * delta_eval_normal + normalized_eval_normal.transpose ());

  Eigen::RowVectorXd term_6th (number_of_neighbors);
  term_6th = 1 - 3 / gaussian_param_sq_ * projected_diff_vec.array ().square ();
  term_6th = 2 * weight.cwiseProduct (term_6th);

  for (int ii = 0; ii < number_of_neighbors; ++ii)
    delta2_g += term_6th (ii) * (delta_eval_normal + delta_eval_normal.transpose ());

  std::vector < Eigen::Matrix3d > temp_v (3);
  for (int z = 0; z < 3; ++z)
    temp_v.at (z).setZero ();
  for (int kk = 0; kk < 3; ++kk)
    for (int ii = 0; ii < 3; ++ii)
      for (int jj = 0; jj < 3; ++jj)
      {
        temp_v.at (kk) (ii, jj) += (4 / (gaussian_param_sq_ * gaussian_param_sq_)
            * weight.cwiseProduct (diff_vec.row (ii)).cwiseProduct (diff_vec.row (jj)).cwiseProduct (neighbor_normals.row (kk))).sum ();
        if (ii == jj)
          temp_v.at (kk) (ii, jj) += (-2 / gaussian_param_sq_ * weight.cwiseProduct (neighbor_normals.row (kk))).sum ();
      }

  for (int ii = 0; ii < number_of_neighbors; ++ii)
  {
    temp_2 = temp_v.at (0) * diff_vec (0, ii) + temp_v.at (1) * diff_vec (1, ii) + temp_v.at (2) * diff_vec (2, ii);
    delta2_g += term_6th (ii) * (Eigen::Matrix3d::Identity () - normalized_eval_normal * normalized_eval_normal.transpose ()) * temp_2 / eval_normal.norm ();
  }

  Eigen::Matrix4d temp_matrix;
  temp_matrix (3, 3) = 0;
  temp_matrix.topLeftCorner<3, 3> () = delta2_g;
  temp_matrix.topRightCorner<3, 1> () = delta_g;
  temp_matrix.bottomLeftCorner<1, 3> () = delta_g.transpose ();

  const float output_gaussian = -temp_matrix.determinant () / std::pow (delta_g.norm (), 4);
  const float output_mean = (delta_g.transpose () * delta2_g * delta_g - std::pow (delta_g.norm (), 2) * delta2_g.trace ()) / std::pow (delta_g.norm (), 3) / 2;

  const float t_sqrt = std::sqrt (output_mean * output_mean - output_gaussian);

  pc1 = output_mean + t_sqrt;
  pc2 = output_mean - t_sqrt;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT> void
pcl::PrincipalCurvaturesEstimationMLS<PointInT, PointNT, PointOutT>::computeFeature (PointCloudOut &output)
{
  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  std::vector<int> nn_indices (k_);
  std::vector<float> nn_dists (k_);

  output.is_dense = true;
  // Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
  if (input_->is_dense)
  {
    // Iterating over the entire index vector
#ifdef _OPENMP
#pragma omp parallel for private(nn_indices, nn_dists) shared(output) num_threads (threads_)
#endif
    for (size_t idx = 0; idx < indices_->size (); ++idx)
    {
      if (this->searchForNeighbors (indices_->at (idx), search_parameter_, nn_indices, nn_dists) == 0)
      {
        output.points[idx].pc1 = output.points[idx].pc2 = std::numeric_limits<float>::quiet_NaN ();
        output.is_dense = false;
        continue;
      }

      // Estimate the principal curvatures at each patch
      computePointPrincipalCurvatures (*normals_, indices_->at (idx), nn_indices, output.points[idx].pc1, output.points[idx].pc2);
    }
  }
  else
  {
    // Iterating over the entire index vector
#ifdef _OPENMP
#pragma omp parallel for private(nn_indices, nn_dists) shared(output) num_threads (threads_)
#endif
    for (size_t idx = 0; idx < indices_->size (); ++idx)
    {
      if (!isFinite (input_->at (indices_->at (idx))) ||
          this->searchForNeighbors (indices_->at (idx), search_parameter_, nn_indices, nn_dists) == 0)
      {
        output.points[idx].pc1 = output.points[idx].pc2 = std::numeric_limits<float>::quiet_NaN ();
        output.is_dense = false;
        continue;
      }

      // Estimate the principal curvatures at each patch
      computePointPrincipalCurvatures (*normals_, (*indices_)[idx], nn_indices, output.points[idx].pc1, output.points[idx].pc2);
    }
  }
}

#define PCL_INSTANTIATE_PrincipalCurvaturesEstimation(T,NT,OutT) template class PCL_EXPORTS pcl::PrincipalCurvaturesEstimation<T,NT,OutT>;
#define PCL_INSTANTIATE_PrincipalCurvaturesEstimationMLS(T,NT,OutT) template class PCL_EXPORTS pcl::PrincipalCurvaturesEstimationMLS<T,NT,OutT>;

#endif    // PCL_FEATURES_IMPL_PRINCIPAL_CURVATURES_H_
