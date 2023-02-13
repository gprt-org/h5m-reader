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

#include "sharedCode.h"
#include "gprt.h"
#include "rng.h"

#define MAX_DEPTH 100

#define EPSILON 2.2204460492503130808472633361816E-16
#define FLT_EPSILON	1.19209290e-7F
#define DBL_EPSILON	2.2204460492503131e-16

float4 over(float4 a, float4 b) {
  float4 result;
  result.a = a.a + b.a * (1.f - a.a);
  result.rgb = (a.rgb * a.a + b.rgb * b.a * (1.f - a.a)) / result.a;
  return result;
}

struct Payload
{
  int2 vol_ids;
  float hitDistance;
};

GPRT_RAYGEN_PROGRAM(DPRayGen, (RayGenData, record))
{
  Payload payload;
  payload.vol_ids = int2(-1, -1);

  uint2 pixelID = DispatchRaysIndex().xy;
  float2 screen = (float2(pixelID) +
                  float2(.5f, .5f)) / float2(record.fbSize);
  const int fbOfs = pixelID.x + record.fbSize.x * pixelID.y;

  RayDesc rayDesc;
  rayDesc.Origin = record.camera.pos;
  rayDesc.Direction =
    normalize(record.camera.dir_00
    + screen.x * record.camera.dir_du
    + screen.y * record.camera.dir_dv
  );
  rayDesc.TMin = 0.0;
  rayDesc.TMax = 10000.0;
  RaytracingAccelerationStructure world = gprt::getAccelHandle(record.world);

  // store double precision ray
  gprt::store(record.dpRays, fbOfs * 2 + 0, double4(rayDesc.Origin.x, rayDesc.Origin.y, rayDesc.Origin.z, rayDesc.TMin));
  gprt::store(record.dpRays, fbOfs * 2 + 1, double4(rayDesc.Direction.x, rayDesc.Direction.y, rayDesc.Direction.z, rayDesc.TMax));

  TraceRay(
    world, // the tree
    RAY_FLAG_NONE, // ray flags
    0xff, // instance inclusion mask
    0, // ray type
    1, // number of ray types
    0, // miss type
    rayDesc, // the ray to trace
    payload // the payload IO
  );

  // generate color using volume ID
  float3 color = normalize(float3(Random(payload.vol_ids.x), Random(payload.vol_ids.x + 1), Random(payload.vol_ids.x + 1)));

  if (payload.vol_ids.x == -2) {
    color = float3(1.0f, 1.0f, 1.0f);
  }

  if (payload.vol_ids.x == -3) {
    color = float3(0.0f, 0.0f, 0.0f);
  }

  gprt::store(record.fbPtr, fbOfs, gprt::make_bgra(color));
}

GPRT_RAYGEN_PROGRAM(SPRayGen, (RayGenData, record))
{
  Payload payload;
  payload.vol_ids = int2(-1, -1);

  uint2 pixelID = DispatchRaysIndex().xy;
  uint2 centerID = DispatchRaysDimensions().xy / 2;
  float2 screen = (float2(pixelID) +
                  float2(.5f, .5f)) / float2(record.fbSize);
  const int fbOfs = pixelID.x + record.fbSize.x * pixelID.y;

  RayDesc rayDesc;
  rayDesc.Origin = record.camera.pos;
  rayDesc.Direction =
    normalize(record.camera.dir_00
    + screen.x * record.camera.dir_du
    + screen.y * record.camera.dir_dv
  );
  rayDesc.TMin = 0.0;
  rayDesc.TMax = 10000.0;

  RaytracingAccelerationStructure world = gprt::getAccelHandle(record.world);

  float4 color = float4(0.f, 0.f, 0.f, 0.f);

  uint32_t maxVolID = record.maxVolID;

  Texture1D colormap = gprt::getTexture1DHandle(record.colormap);
  SamplerState sampler = gprt::getSamplerHandle(record.colormapSampler);

  for (int i = 0; i < MAX_DEPTH; ++i) {
    TraceRay(
      world, // the tree
      RAY_FLAG_NONE, // RAY_FLAG_CULL_BACK_FACING_TRIANGLES, // ray flags
      0xff, // instance inclusion mask
      0, // ray type
      1, // number of ray types
      0, // miss type
      rayDesc, // the ray to trace
      payload // the payload IO
    );

    // float3 color = normalize(float3(Random(payload.vol_ids.x), Random(payload.vol_ids.x + 1), Random(payload.vol_ids.x + 1)));

    if (payload.vol_ids.x == -2) {
      color = over(color, float4(0.0f, 0.0f, 0.0f, 1.0f));
      break;
    }

    else if (payload.vol_ids.x == -3) {
      color = over(color, float4(0.0f, 0.0f, 0.0f, 1.f));
      break;
    }

    else if (payload.vol_ids.x == record.complementID) {
      rayDesc.Origin = rayDesc.Origin + rayDesc.Direction * (payload.hitDistance + .01);
      continue;
    }
    else if (payload.vol_ids.x == record.graveyardID) {
      rayDesc.Origin = rayDesc.Origin + rayDesc.Direction * (payload.hitDistance + .01);
      continue;
    }

    else {
      float dataValue = float(payload.vol_ids.x) / float(maxVolID);
      float4 xf = colormap.SampleGrad(sampler, dataValue, 0.f, 0.f);

      if (xf.w != 0.f) {
        color = over(color, xf);
        if (color.a > .99) break;
      }

      rayDesc.Origin = rayDesc.Origin + rayDesc.Direction * (payload.hitDistance + .01);
    }

  }
  
  // if (all(pixelID == centerID))
  //   printf("center vol ID is front - %d back - %d\n", payload.vol_ids.x, payload.vol_ids.y);

  // a crosshair
  if (any(pixelID == centerID)) color = float4(1.f, 1.f, 1.f, 1.f) - color;

  // Composite on top of everything else our user interface
  Texture2D texture = gprt::getTexture2DHandle(record.guiTexture);
  SamplerState guiSampler = gprt::getDefaultSampler();

  float4 guiColor = texture.SampleGrad(guiSampler, screen, float2(0.f, 0.f), float2(0.f, 0.f));

  float4 finalColor = over(guiColor, float4(color.r, color.g, color.b, 1.f));

  gprt::store(record.fbPtr, fbOfs, gprt::make_bgra(finalColor));
}



struct Sampler {
  RaytracingAccelerationStructure mesh;

  int2 operator()(float3 coordinate) {
    Payload payload;
    payload.vol_ids = int2(-1, -1);

    RayDesc rayDesc;
    rayDesc.Origin = coordinate;
    rayDesc.Direction = float3(0.f,0.f,1.f);
    rayDesc.TMin = 0.0;
    rayDesc.TMax = 10000.0;

    TraceRay(
      mesh, // the tree
      RAY_FLAG_NONE, // ray flags
      0xff, // instance inclusion mask
      0, // ray type
      1, // number of ray types
      0, // miss type
      rayDesc, // the ray to trace
      payload // the payload IO
    );

    return payload.vol_ids;
  }
};

GPRT_RAYGEN_PROGRAM(SPVolVis, (RayGenData, record))
{
  Payload payload;
  payload.vol_ids = int2(-1, -1);
  uint2 pixelID = DispatchRaysIndex().xy;
  uint2 centerID = DispatchRaysDimensions().xy / 2;
  float2 screen = (float2(pixelID) +
                  float2(.5f, .5f)) / float2(record.fbSize);
  const int fbOfs = pixelID.x + record.fbSize.x * pixelID.y;

  int frameId = record.frameID; // todo, change per frame
  LCGRand rng = get_rng(frameId, DispatchRaysIndex().xy, DispatchRaysDimensions().xy);

  RayDesc rayDesc;
  rayDesc.Origin = record.camera.pos;
  rayDesc.Direction =
    normalize(record.camera.dir_00
    + screen.x * record.camera.dir_du
    + screen.y * record.camera.dir_dv
  );
  rayDesc.TMin = 0.0;
  rayDesc.TMax = 10000.0;

  // typical ray AABB intersection test
  float3 dirfrac;    // direction is unit direction vector of ray
  dirfrac.x = 1.0f / rayDesc.Direction.x;
  dirfrac.y = 1.0f / rayDesc.Direction.y;
  dirfrac.z = 1.0f / rayDesc.Direction.z;
  // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
  // origin is origin of ray
  float3 rt = record.aabbMax;
  float t2 = (rt.x - rayDesc.Origin.x)*dirfrac.x;
  float t4 = (rt.y - rayDesc.Origin.y)*dirfrac.y;
  float t6 = (rt.z - rayDesc.Origin.z)*dirfrac.z;
  float3 lb = record.aabbMin;
  float t1 = (lb.x - rayDesc.Origin.x)*dirfrac.x;
  float t3 = (lb.y - rayDesc.Origin.y)*dirfrac.y;
  float t5 = (lb.z - rayDesc.Origin.z)*dirfrac.z;
  float thit0 = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
  float thit1 = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));    // clip hit to near position
  // thit0 = max(thit0, rayDesc.TMin);
  // thit1 = min(thit1, rayDesc.TMax);
  // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
  bool hit = true;
  if (thit1 < 0) { hit = false; }
  // if tmin > tmax, ray doesn't intersect AABB
  if (thit0 >= thit1) { hit = false; }

  Sampler meshSampler;
  meshSampler.mesh = gprt::getAccelHandle(record.world);

  float4 color = float4(0.f, 0.f, 0.f, 1.f);
  if (hit) {
    // float t = .001f;
    // float absorbance = 1.f - exp(-(thit1 - thit0) * t);

    // if (all(pixelID == centerID)) {
    // // printf("aabb %f %f %f aabbmax %f %f %f \n", record.aabbMin.x, record.aabbMin.y, record.aabbMin.z, record.aabbMax.x, record.aabbMax.y, record.aabbMax.z);

    // // printf("t1 %f t2 %f t3 %f t4 %f t5 %f t6 %f \n", t1, t2, t3, t4, t5, t6);
    // // printf("thit0 %f thit1 %f\n", thit0, thit1);
    //   printf("dist %f t %f\n", thit1 - thit0, absorbance);
    // }

    // color = float4(1.f, 0.f, 0.f, 1.f) * absorbance;

    float unit = record.unit;
    float majorantExtinction = 1.f; // todo, DDA or something similar
    float t = thit0;
    uint32_t numVolumes = record.numVolumes;
    uint32_t maxVolID = record.maxVolID;

    Texture1D colormap = gprt::getTexture1DHandle(record.colormap);
    SamplerState sampler = gprt::getSamplerHandle(record.colormapSampler);

    // while (true) {
    for (int i = 0; i < MAX_DEPTH; ++i) {

      // Sample a distance
      t = t - (log(1.0f - lcg_randomf(rng)) / majorantExtinction) * unit;

      // A boundary has been hit
      if (t >= thit1) break;

      // Update current position
      float3 x = rayDesc.Origin + t * rayDesc.Direction;

      // Sample heterogeneous media
      float dataValue = float(meshSampler(x).y); // we want the ID of what's on our side

      if (dataValue == float(record.complementID)) continue;
      if (dataValue == float(record.graveyardID)) continue;

      dataValue = dataValue / float(maxVolID);
      float4 xf = colormap.SampleGrad(sampler, dataValue, 0.f, 0.f);
      //   float remapped1 = (dataValue - volDomain.lower) / (volDomain.upper - volDomain.lower);
      //   float remapped2 = (remapped1 - xfDomain.lower) / (xfDomain.upper - xfDomain.lower);
      //   xf = tex2D<float4>(lp.transferFunc.texture,remapped2,0.5f);
      //   xf.w *= lp.transferFunc.opacityScale;
      // }

      // Check if an emission occurred
      if (lcg_randomf(rng) < xf.w / (majorantExtinction)) {
      //   prd.tHit = min(prd.tHit, t);
      //   prd.rgba = vec4f(vec3f(xf), 1.f);
        color = float4(xf.rgb, 1.f);
        break;
      }
    }


    // RaytracingAccelerationStructure world = gprt::getAccelHandle(record.world);

    // TraceRay(
    //   world, // the tree
    //   RAY_FLAG_CULL_BACK_FACING_TRIANGLES, // ray flags
    //   0xff, // instance inclusion mask
    //   0, // ray type
    //   1, // number of ray types
    //   0, // miss type
    //   rayDesc, // the ray to trace
    //   payload // the payload IO
    // );

    // float3 color = normalize(float3(Random(payload.vol_id), Random(payload.vol_id + 1), Random(payload.vol_id + 1)));

    // if (payload.vol_id == -2) {
    //   color = float3(1.0f, 1.0f, 1.0f);
    // }

    // if (payload.vol_id == -3) {
    //   color = float3(0.0f, 0.0f, 0.0f);
    // }

  }


  // Texture1D colormap = gprt::getTexture1DHandle(record.colormap);
  // SamplerState sampler = gprt::getDefaultSampler();

  // color = colormap.SampleGrad(sampler, float(pixelID.x) / float(DispatchRaysDimensions().x), 0.f, 0.f);

  float4 prevColor = gprt::load<float4>(record.accumPtr, fbOfs);
  float4 finalColor = (1.f / float(frameId)) * color + (float(frameId - 1) / float(frameId)) * prevColor;
  gprt::store<float4>(record.accumPtr, fbOfs, finalColor);

  // Composite on top of everything else our user interface
  Texture2D texture = gprt::getTexture2DHandle(record.guiTexture);
  SamplerState sampler = gprt::getDefaultSampler();
  float4 guiColor = texture.SampleGrad(sampler, screen, float2(0.f, 0.f), float2(0.f, 0.f));
  finalColor = over(guiColor, float4(finalColor.r, finalColor.g, finalColor.b, finalColor.a));
  gprt::store(record.fbPtr, fbOfs, gprt::make_bgra(finalColor));
}

GPRT_MISS_PROGRAM(miss, (MissProgData, record), (Payload, payload))
{
  uint2 pixelID = DispatchRaysIndex().xy;
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  payload.vol_ids = int2((pattern & 1) ? -2 : -3, (pattern & 1) ? -2 : -3);
}

struct DPAttribute
{
  double2 bc;
};

GPRT_COMPUTE_PROGRAM(DPTriangle, (DPTriangleData, record), (1,1,1))
{
  int primID = DispatchThreadID.x;
  int3 indices = gprt::load<int3>(record.index, primID);
  double3 A = gprt::load<double3>(record.vertex, indices.x);
  double3 B = gprt::load<double3>(record.vertex, indices.y);
  double3 C = gprt::load<double3>(record.vertex, indices.z);
  double3 dpaabbmin = min(min(A, B), C);
  double3 dpaabbmax = max(max(A, B), C);
  float3 fpaabbmin = float3(dpaabbmin) - float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON); // todo, round this below smallest float
  float3 fpaabbmax = float3(dpaabbmax) + float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON); // todo, round this below smallest float
  gprt::store(record.aabbs, 2 * primID + 0, fpaabbmin);
  gprt::store(record.aabbs, 2 * primID + 1, fpaabbmax);
}

GPRT_CLOSEST_HIT_PROGRAM(DPTriangle, (DPTriangleData, record), (Payload, payload), (DPAttribute, attribute))
{
  uint hit_kind = HitKind();
  if (hit_kind == HIT_KIND_TRIANGLE_FRONT_FACE){
    payload.vol_ids.x = record.vols[0];
    payload.vol_ids.y = record.vols[1];
  }
  else {
    payload.vol_ids.x = record.vols[1];
    payload.vol_ids.y = record.vols[0];
  }
}

double3 dcross (in double3 a, in double3 b) { return double3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }

float next_after(float a) {
  uint a_ = asuint(a);
  if (a < 0) {
    a_--;
  } else {
    a_++;
  }
  return asfloat(a_);
}

/* Function to return the vertex with the lowest coordinates. To force the same
    ray-edge computation, the Plücker test needs to use consistent edge
    representation. This would be more simple with MOAB handles instead of
    coordinates... */
inline bool first( in double3 a, in double3 b )
{
  if(a.x < b.x) return true;

  if (a.x == b.x && a.y < b.y) return true;

  if (a.y == b.y && a.z < b.z) return true;

  return false;
}

double plucker_edge_test( in double3 vertexa, in double3 vertexb, in double3 ray, in double3 ray_normal )
{
  double pip;
  const double near_zero = 10 * DBL_EPSILON;

  if( first( vertexa, vertexb ) )
  {
      double3 edge        = vertexb - vertexa;
      double3 edge_normal = dcross(edge, vertexa);
      pip                 = dot(ray, edge_normal) + dot(ray_normal, edge);
  }
  else
  {
      double3 edge        = vertexa - vertexb;
      double3 edge_normal = dcross(edge, vertexb);
      pip                = dot(ray, edge_normal) + dot(ray_normal, edge);
      pip                = -pip;
  }

  if( near_zero > abs( pip ) ) pip = 0.0;

  return pip;
}

GPRT_INTERSECTION_PROGRAM(DPTrianglePlucker, (DPTriangleData, record))
{

  uint2 pixelID = DispatchRaysIndex().xy;
  uint2 dims = DispatchRaysDimensions().xy;
  bool debug = false;
  if ((pixelID.x == dims.x / 2) && (pixelID.y == dims.y / 2)) debug = true;

  uint flags = RayFlags();


  // Just skip if we for some reason cull both...
  if ( ((flags & RAY_FLAG_CULL_BACK_FACING_TRIANGLES) != 0) &&
       ((flags & RAY_FLAG_CULL_FRONT_FACING_TRIANGLES) != 0)) return;

  bool useOrientation = false;
  int orientation = 0;
  if ((flags & RAY_FLAG_CULL_BACK_FACING_TRIANGLES) != 0) {
    orientation = -1;
    useOrientation = true;
  }
  else if ((flags & RAY_FLAG_CULL_FRONT_FACING_TRIANGLES) != 0) {
    orientation = 1;
    useOrientation = true;
  }

  int primID = PrimitiveIndex();
  int3 indices = gprt::load<int3>(record.index, primID);
  double3 v0 = gprt::load<double3>(record.vertex, indices.x);
  double3 v1 = gprt::load<double3>(record.vertex, indices.y);
  double3 v2 = gprt::load<double3>(record.vertex, indices.z);

//  uint2 pixelID = DispatchRaysIndex().xy;
  const int fbOfs = pixelID.x + record.fbSize.x * pixelID.y;
  double4 raydata1 = gprt::load<double4>(record.dpRays, fbOfs * 2 + 0);
  double4 raydata2 = gprt::load<double4>(record.dpRays, fbOfs * 2 + 1);
  double3 origin = double3(raydata1.x, raydata1.y, raydata1.z);//ObjectRayOrigin();
  double3 direction = double3(raydata2.x, raydata2.y, raydata2.z);//ObjectRayDirection();
  double tMin = raydata1.w;
  double tCurrent = raydata2.w;

  const double3 raya = direction;
  const double3 rayb = dcross(direction, origin);

  // Determine the value of the first Plucker coordinate from edge 0
  double plucker_coord0 = plucker_edge_test(v0, v1, raya, rayb);

  // If orientation is set, confirm that sign of plucker_coordinate indicate
  // correct orientation of intersection
  if( useOrientation && orientation * plucker_coord0 > 0 ) {
    return;
  }

  // Determine the value of the second Plucker coordinate from edge 1
  double plucker_coord1 = plucker_edge_test( v1, v2, raya, rayb );

  // If orientation is set, confirm that sign of plucker_coordinate indicate
  // correct orientation of intersection
  if( useOrientation &&  orientation * plucker_coord1 > 0) return;

  // If the orientation is not specified, all plucker_coords must be the same sign or
  // zero.
  else if( ( 0.0 < plucker_coord0 && 0.0 > plucker_coord1 ) || ( 0.0 > plucker_coord0 && 0.0 < plucker_coord1 ) ) return;

  // Determine the value of the second Plucker coordinate from edge 2
  double plucker_coord2 = plucker_edge_test( v2, v0, raya, rayb );

  // If orientation is set, confirm that sign of plucker_coordinate indicate
  // correct orientation of intersection
  if( useOrientation && orientation * plucker_coord2 > 0) return;
  // If the orientation is not specified, all plucker_coords must be the same sign or
  // zero.
  else if( ( 0.0 < plucker_coord1 && 0.0 > plucker_coord2 ) || ( 0.0 > plucker_coord1 && 0.0 < plucker_coord2 ) ||
           ( 0.0 < plucker_coord0 && 0.0 > plucker_coord2 ) || ( 0.0 > plucker_coord0 && 0.0 < plucker_coord2 ) )
  {
    return; // EXIT_EARLY;
  }

  // check for coplanar case to avoid dividing by zero
  if( 0.0 == plucker_coord0 && 0.0 == plucker_coord1 && 0.0 == plucker_coord2 ) {
    return; // EXIT_EARLY;
  }

  // get the distance to intersection
  const double inverse_sum = 1.0 / ( plucker_coord0 + plucker_coord1 + plucker_coord2 );
  const double3 intersection = double3( plucker_coord0 * inverse_sum * v2 +
                                        plucker_coord1 * inverse_sum * v0 +
                                        plucker_coord2 * inverse_sum * v1 );

  // To minimize numerical error, get index of largest magnitude direction.
  int idx            = 0;
  double max_abs_dir = 0;
  for( unsigned int i = 0; i < 3; ++i )
  {
      if( abs( direction[i] ) > max_abs_dir )
      {
          idx         = i;
          max_abs_dir = abs( direction[i] );
      }
  }
  const double dist = ( intersection[idx] - origin[idx] ) / direction[idx];


  double t = dist;
  double u = plucker_coord2 * inverse_sum;
  double v = plucker_coord0 * inverse_sum;

  if( u < 0.0 || v < 0.0 || (u+v) > 1.0 ) t = -1.0;

  if (t > tCurrent) return;
  if (t < tMin) return;

  // update current double precision thit
  gprt::store<double>(record.dpRays, fbOfs * 8 + 7, t);

  DPAttribute attr;
  attr.bc = double2(u, v);
  float f32t = float(t);
  if (double(f32t) < t) f32t = next_after(f32t);

  // compute the triangle normal
  double3 norm = dcross(double3(v1 - v0), double3(v2 - v0));

  uint hit_kind;

  if (dot(norm, direction) > 0 ) {
    hit_kind = HIT_KIND_TRIANGLE_BACK_FACE;
  } else {
    hit_kind = HIT_KIND_TRIANGLE_FRONT_FACE;
  }

  ReportHit(f32t, hit_kind, attr);
}

struct SPAttribute
{
  float2 bc;
};

GPRT_CLOSEST_HIT_PROGRAM(SPTriangle, (SPTriangleData, record), (Payload, payload), (SPAttribute, attribute))
{
  uint hit_kind = HitKind();
  if (hit_kind == HIT_KIND_TRIANGLE_FRONT_FACE) {
    payload.vol_ids.x = record.vols[0];
    payload.vol_ids.y = record.vols[1];
  }
  else {
    payload.vol_ids.x = record.vols[1];
    payload.vol_ids.y = record.vols[0];
  }
  payload.hitDistance = RayTCurrent();
}