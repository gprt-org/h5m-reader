#include "sharedCode.h"
#include "dda.hlsli"

GPRT_RAYGEN_PROGRAM(SPVoxelize, (RayGenData, record))
{
    uint3 voxelID = DispatchRaysIndex().xyz;
    float3 worldAABBMin = record.aabbMin;
    float3 worldAABBMax = record.aabbMax;

    uint3 gridDims = record.gridDims;
    
    float3 worldPos = gridPosToWorld(
        float3(voxelID) + .5, 
        worldAABBMin, worldAABBMax, gridDims
    );

    RaytracingAccelerationStructure world = gprt::getAccelHandle(record.world);
    world = gprt::getAccelHandle(record.world);   

    Payload payload;
    payload.vol_ids = int2(-1, -1);

    RayDesc rayDesc;
    rayDesc.Origin = worldPos;
    rayDesc.Direction = float3(0.f, 0.f, 1.f); 
    rayDesc.TMin = 0.f;
    rayDesc.TMax = 1e20f;

    TraceRay(
        world, // the tree
        RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, // ray flags
        0xff, // instance inclusion mask
        0, // ray type
        gprt::getNumRayTypes(), // number of ray types
        0, // miss type
        rayDesc, // the ray to trace
        payload // the payload IO
    );

    // store results into the grid.
    // demonstrating for the moment an atomic add, but not really necessary...
    gprt::atomicAdd32f(record.ddaGrid,
        voxelID.x + voxelID.y * gridDims.x + voxelID.z * gridDims.x * gridDims.y,
        payload.vol_ids.x // store the volume we're escaping from
    );
}

GPRT_RAYGEN_PROGRAM(DPVoxelize, (RayGenData, record))
{

}