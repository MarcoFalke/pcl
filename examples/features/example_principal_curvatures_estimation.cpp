/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2009-2011, Willow Garage, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 * * Neither the name of Willow Garage, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */


#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>


int
main (int, char** argv)
{
  std::string filename = argv[1];
  std::cout << "Reading " << filename << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (filename, *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file");
    return (-1);
  }

  std::cout << "Loaded " << cloud->points.size () << " points." << std::endl;

  // Compute the normals
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
  normal_estimation.setInputCloud (cloud);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  normal_estimation.setSearchMethod (tree);

  pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::Normal>);

  normal_estimation.setRadiusSearch (0.03);

  normal_estimation.compute (*cloud_with_normals);

  // Setup the principal curvatures computation:
  // PCA
  pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;
  // MLS
  pcl::PrincipalCurvaturesEstimationMLS<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation_mls;

  // Provide the original point cloud (without normals)
  principal_curvatures_estimation.setInputCloud (cloud);
  principal_curvatures_estimation_mls.setInputCloud (cloud);

  // Provide the point cloud with normals
  principal_curvatures_estimation.setInputNormals (cloud_with_normals);
  principal_curvatures_estimation_mls.setInputNormals (cloud_with_normals);

  // Use the same KdTree from the normal estimation
  principal_curvatures_estimation.setSearchMethod (tree);
  principal_curvatures_estimation.setRadiusSearch (1.0);
  principal_curvatures_estimation_mls.setSearchMethod (tree);
  principal_curvatures_estimation_mls.setKSearch (90);
  principal_curvatures_estimation_mls.setGaussianParam(0.09);
#ifdef _OPENMP
  principal_curvatures_estimation_mls.setNumberOfThreads(0); // 0 for auto
#endif

  // Actually compute the principal curvatures
  pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures_pca (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
  principal_curvatures_estimation.compute (*principal_curvatures_pca);
  pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures_mls (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
  principal_curvatures_estimation_mls.compute (*principal_curvatures_mls);

  std::cout << "output pca eigenval curvature points.size (): " << principal_curvatures_pca->points.size () << std::endl;
  std::cout << "output mls surface curvature points.size (): " << principal_curvatures_mls->points.size () << std::endl;

  // Display and retrieve the shape context descriptor vector for the 0th point.
  pcl::PrincipalCurvatures descriptor_evs = principal_curvatures_pca->points[0];
  std::cout << descriptor_evs << std::endl;
  pcl::PrincipalCurvatures descriptor_mls = principal_curvatures_mls->points[0];
  std::cout << descriptor_mls << std::endl;

  return 0;
}
