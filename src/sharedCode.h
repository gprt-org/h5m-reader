// MIT License

// Copyright (c) 2022 Nathan V. Morrical

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "gprt.h"

/* variables for the triangle mesh geometry */
struct DPTriangleData
{
  /*! array/buffer of vertex indices */
  alignas(16) gprt::Buffer index; // vec3f*
  /*! array/buffer of vertex positions */
  alignas(16) gprt::Buffer vertex; // float *
  /*! array/buffer of AABBs */
  alignas(16) gprt::Buffer aabbs;


  /*! array/buffer of double precision rays */
  alignas(16) gprt::Buffer dpRays;
  alignas(8) int2 fbSize;
  /*! volume on the forward and reverse side of the surface */
  alignas(8) int2 vols;
};

struct SPTriangleData
{
  /*! array/buffer of vertex indices */
  alignas(16) gprt::Buffer index; // vec3f*
  /*! array/buffer of vertex positions */
  alignas(16) gprt::Buffer vertex; // float *
  /*! volume on the forward and reverse side of the surface */
  alignas(8) int2 vols;
};

struct RayGenData
{
  alignas(16) gprt::Buffer accumPtr;
  alignas(16) gprt::Buffer fbPtr;
  alignas(16) gprt::Buffer dpRays;

  alignas(16) gprt::Texture guiTexture;

  // a relative unit for delta tracking, similar to "dt"
  alignas(4) float unit;
  alignas(4) uint32_t frameID;
  alignas(4) uint32_t numVolumes;
  alignas(4) uint32_t maxVolID;
  alignas(4) uint32_t graveyardID;
  alignas(4) uint32_t complementID;
  // colormap for visualization
  alignas(16) gprt::Texture colormap;
  alignas(16) gprt::Sampler colormapSampler;

  alignas(8) int2 fbSize;

  alignas(16) gprt::Accel world;
  alignas(16) gprt::Buffer partTrees; // gprt::Accel*.

  alignas(16) float3 aabbMin;
  alignas(16) float3 aabbMax;

  struct {
    alignas(16) float3 pos;
    alignas(16) float3 dir_00;
    alignas(16) float3 dir_du;
    alignas(16) float3 dir_dv;
  } camera;
};

/* variables for the miss program */
struct MissProgData
{
  alignas(16) float3  color0;
  alignas(16) float3  color1;
};
