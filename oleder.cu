#include "hip/hip_runtime.h"
// --------------------------------------------------------------------//
//                                                                     //
// title                  :gridding.cu                                 //
// description            :Sort and Gridding process.                  //
// author                 :                                            //
//                                                                     //
// --------------------------------------------------------------------//

#include <boost/sort/sort.hpp>
#include "gridding.h"
using boost::sort::block_indirect_sort;

double *d_lons;
double *d_lats;
double *d_data;
double *d_weights;
uint64_t *d_hpx_idx;
uint32_t *d_start_ring;
//texture<uint32_t> tex_start_ring;
__constant__ uint32_t d_const_zyx[3];
uint32_t *d_zyx;
double *d_xwcs;
double *d_ywcs;
double *d_datacube;
double *d_weightscube;
__constant__ double d_const_kernel_params[3];
__constant__ GMaps d_const_GMaps;

/* Print a array pair. */
void print_double_array(double *A, double *B, uint32_t num){
    printf("Array (A, B) = <<<\n");
    for(int i=0; i<10; ++i){
        printf("(%f, %f), ", A[i], B[i]);
    }
    printf("\n..., \n");
    for(int i=num-11; i<num; ++i){
        printf("(%f, %f), ", A[i], B[i]);
    }
    printf("\n>>>\n\n");
}


/*********************************Sort input points with CPU*******************************/
/**
 * @brief   Sort input points with CPU and Create two level lookup table.
 * @param   sort_param: set the sort parameters for the chosen type.
 * */
void init_input_with_cpu(const int &sort_param) {
    double iTime1 = cpuSecond();
    uint32_t data_shape = h_GMaps.data_shape;
    std::vector<HPX_IDX> V(data_shape);
    V.reserve(data_shape);

    // Get HEALPix index and input index of each input point.
    for(int i=0; i < data_shape; ++i) {
        double theta = HALFPI - DEG2RAD * h_lats[i];
        double phi = DEG2RAD * h_lons[i];
        uint64_t hpx = h_ang2pix(theta, phi);
        V[i] = HPX_IDX(hpx, i);             // (HEALPix_index, input_index)
    }

    // Sort input points by param (key-value sort). KEY: HEALPix index VALUE: array index
    double iTime2 = cpuSecond();
    if (sort_param == BLOCK_INDIRECT_SORT) {
        boost::sort::block_indirect_sort(V.begin(), V.end());
    } else if (sort_param == PARALLEL_STABLE_SORT) {
        boost::sort::parallel_stable_sort(V.begin(), V.end());
    } else if (sort_param == STL_SORT) {
        std::sort(V.begin(), V.end());
    }
    double iTime3 = cpuSecond();

    // Copy the HEALPix, lons, lats and data for sorted input points
    // Sort the input points according the sorted input index.
    h_hpx_idx = RALLOC(uint64_t, data_shape + 1);
    for(int i=0; i < data_shape; ++i){
        h_hpx_idx[i] = V[i].hpx;
    }
    h_hpx_idx[data_shape] = h_Healpix._npix;
    double *tempArray = RALLOC(double, data_shape);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lons[V[i].inx];
    }
    swap(h_lons, tempArray);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_lats[V[i].inx];
    }
    swap(h_lats, tempArray);
    for(int i=0; i < data_shape; ++i){
        tempArray[i] = h_data[V[i].inx];
    }
    swap(h_data, tempArray);
    DEALLOC(tempArray);

    // Pre-process by h_hpx_idx
    double iTime4 = cpuSecond();
    uint64_t first_ring = h_pix2ring(h_hpx_idx[0]); // HEALPix 划分后的HEALPix区块的首行
    uint32_t temp_count = (uint32_t)(1 + h_pix2ring(h_hpx_idx[data_shape - 1]) - first_ring);   // The total number of ring
    h_Healpix.firstring = first_ring;
    h_Healpix.usedrings = temp_count;
    h_start_ring = RALLOC(uint32_t, temp_count + 1); // 环号下标 ——> 区块号下标
    h_start_ring[0] = 0;    // 首行(的下标0）——> 起始区块号（的下标，也就是第一个区块号0）
    uint64_t startpix, num_pix_in_ring;   // important
    uint32_t ring_idx = 0;
    bool shifted;
    for(uint64_t cnt_ring = 1; cnt_ring < temp_count; ++cnt_ring) { // 从第二行开始
        h_get_ring_info_small(cnt_ring + first_ring, startpix, num_pix_in_ring, shifted);
        uint32_t cnt_ring_idx = searchLastPosLessThan(h_hpx_idx, ring_idx, data_shape, startpix);   // Binary search the startpix index
        if (cnt_ring_idx == data_shape) {
            cnt_ring_idx = ring_idx - 1;
        }
        ring_idx = cnt_ring_idx + 1;    // The start array index of the first HEALPix index in one ring.
        h_start_ring[cnt_ring] = ring_idx;  // Construct the R_start  行号：起始区块号
    }
    h_start_ring[temp_count] = data_shape;  // 这里貌似有问题，最大行的其实区块这里直接赋值为最大index
    double iTime5 = cpuSecond();

    // Release
    vector<HPX_IDX> vtTemp;
    vtTemp.swap(V);

    // Get runtime
    double iTime6 = cpuSecond();
    printf("%f, ", (iTime6 - iTime1) * 1000.);
//    printf("%f, %f, %f\n", (iTime3 - iTime2) * 1000., (iTime5 - iTime4) * 1000., (iTime6 - iTime1) * 1000.);
}


/*********************************Sort input points with GPU*******************************/
/**
 * @brief   Sort input points with GPU and Create two level lookup table.
 * @param   sort_param: set the sort parameters for the chosen type.
 * @note    Through our testing, the performance of sort input points with
 *          CPU is higher than its in GPU, here we provide the GPU sort
 *          interface for reference.
 * */

/* Initialize output spectrals and weights. */
void init_output(){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    h_datacube = RALLOC(double, num);
    h_weightscube = RALLOC(double, num);
    for(uint32_t i = 0; i < num; ++i){
        h_datacube[i] = 0.;
        h_weightscube[i] = 0.;
    }
}

/* Sinc function with simple singularity check. */
__device__ double sinc(double x){
    if(fabs(x) < 1.e-10)
        return 1.;
    else
        return sin(x) / x;
}

/* Grid-kernel definitions. */
__device__ double kernel_func_ptr(double distance, double bearing){
    if(d_const_GMaps.kernel_type == GAUSS1D){   // GAUSS1D
//        if(distance <= 0.01)
//            return 1;
//        else
//            return 0;
        return exp(-distance * distance * d_const_kernel_params[0]);
    }
    else if(d_const_GMaps.kernel_type == GAUSS2D){  // GAUSS2D
        double ellarg = (\
                pow(d_const_kernel_params[0], 2.0)\
                    * pow(sin(bearing - d_const_kernel_params[2]), 2.0)\
                + pow(d_const_kernel_params[1], 2.0)\
                    * pow(cos(bearing - d_const_kernel_params[2]), 2.0));
        double Earg = pow(distance / d_const_kernel_params[0] /\
                       d_const_kernel_params[1], 2.0) / 2. * ellarg;
        return exp(-Earg);
    }
    else if(d_const_GMaps.kernel_type == TAPERED_SINC){ // TAPERED_SINC
        double arg = PI * distance / d_const_kernel_params[0];
        return sinc(arg / d_const_kernel_params[2])\
            * exp(pow(-(arg / d_const_kernel_params[1]), 2.0));
    }
}

/* Binary search key in hpx_idx array. */
__host__ __device__ uint32_t searchLastPosLessThan(uint64_t *values, uint32_t left, uint32_t right, uint64_t _key){
    if(right <= left)
        return right;
    uint32_t low = left, mid, high = right - 1;
    while (low < high){
        mid = low + ((high - low + 1) >> 1);
        if (values[mid] < _key)
            low = mid;
        else
            high = mid - 1;
    }
    if(values[low] < _key)
        return low;
    else
        return right;
}


/********************************************HCGrid****************************************/
/**
 * @brief   Execute gridding in GPU.
 * @param   d_lons: longitude
 * */
__global__ void hcgrid (
        double *d_lons,
        double *d_lats,
        double *d_data,
        double *d_weights,
        double *d_xwcs,
        double *d_ywcs,
        double *d_datacube,
        double *d_weightscube,
        uint32_t *d_start_ring,
        uint64_t *d_hpx_idx) {
    uint32_t warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // warp index of the whole 1Dim grid and 1Dim block.
    uint32_t thread_id = ((warp_id % d_const_GMaps.block_warp_num) * 32 + threadIdx.x % 32) * d_const_GMaps.factor;   // thread index in one ring.
    int get_num = 0;
    int target_num = 0;
    if (thread_id < d_const_zyx[1]) {
        uint32_t left = thread_id;    //Initial left
        uint32_t right = left + d_const_GMaps.factor - 1;   // Initial right
        if (right > d_const_zyx[1]) {                      // 这块儿感觉有问题，怎么是等于每行的最大索引时-1了？(待定）
            right = d_const_zyx[1];
        }
        uint32_t step = (warp_id / d_const_GMaps.block_warp_num) * d_const_zyx[1];    // Thread step for change the ring
        left = left + step;
        right = right + step;
        double temp_weights[3], temp_data[3], l1[3], b1[3];  //这里预设为最大线程粗化因子为3，所以每次连续写入factor个值
        for (thread_id = left; thread_id <= right; ++thread_id) {
            temp_weights[thread_id - left] = d_weightscube[thread_id];
            temp_data[thread_id - left] = d_datacube[thread_id];
            l1[thread_id - left] = d_xwcs[thread_id] * DEG2RAD;
            b1[thread_id - left] = d_ywcs[thread_id] * DEG2RAD;
        }
//        if((l1[0] * RAD2DEG == 180) && (b1[0] * RAD2DEG == 0)){
//            ++target_num;
//            printf("target_num=%d\n", target_num);
//        }

        // get northeast ring and southeast ring
        double disc_theta = HALFPI - b1[0];     // disc中心点所在行的赤纬
        double disc_phi = l1[0];                // disc中心点所在行的赤经
        double utheta = disc_theta - d_const_GMaps.disc_size;   // 最上面一行中心点的赤纬
        double north_theta = utheta * RAD2DEG;
        if (utheta * RAD2DEG < 0){
            utheta = 0;
        }  // 这里修改了，影响极点位置
//        printf("utheta = %f\n", utheta*RAD2DEG);
        uint64_t upix = d_ang2pix(utheta, disc_phi);            //最上面一行中心点所属的HEALPix的pixel索引
        uint64_t uring = d_pix2ring(upix);                      //最上面一行中心点所在行的HEALPix的行号
//        printf("uring = %d\n", d_const_Healpix.firstring);
        if (uring < d_const_Healpix.firstring){
            uring = d_const_Healpix.firstring;
        }
        utheta = disc_theta + d_const_GMaps.disc_size;  // 最下面一行中心点的赤纬
//        printf("utheta = %f\n", utheta*RAD2DEG);
        upix = d_ang2pix(utheta, disc_phi);             // 最下面一行中心点所属的HEALPix的pixel索引
        uint64_t dring = d_pix2ring(upix);              // 最下面一行中心点所在行的HEALPix行号
//        printf("dring = %d\n", dring);
        if (dring >= d_const_Healpix.firstring + d_const_Healpix.usedrings){
            dring = d_const_Healpix.firstring + d_const_Healpix.usedrings - 1;
        }
//        else if(dring < North_ring){
//            dring = North_ring;
//        }
//        printf("uring = %d\n", uring);
//        printf("dring = %d\n", dring);
//        double z, z0, cosrbig, xa, x, ysq, dphi;
//        z0 = cos(disc_theta);
//        cosbig = cos(disc_size);
//        xa = 1 / sqrt((1 - z0) * (1 + z0));

        // Go from the Northeast ring to the Southeast one
        uint32_t start_int = d_start_ring[uring - d_const_Healpix.firstring];  // get the first HEALPix index
        // tex1Dfetch(tex_start_ring, uring - d_const_Healpix.firstring);
        while (uring <= dring) {                                                            // of one ring.
            // get ring info
            uint32_t end_int = d_start_ring[uring - d_const_Healpix.firstring+1];
                    // tex1Dfetch(tex_start_ring, uring - d_const_Healpix.firstring+1);
            uint64_t startpix, num_pix_in_ring, mid_pixel;
            bool shifted;
            d_get_ring_info_small(uring, startpix, num_pix_in_ring, shifted);
            double utheta, uphi, hpx_zero_theta, hpx_zero_phi, rc_theta, rc_phi;
            double d;
            d_pix2ang(startpix, utheta, uphi);

            // double start_theta, start_phi;
            // start_theta = utheta;
            // start_phi = uphi;
            // get lpix and rpix
//            upix = d_ang2pix(HALFPI-b1[0], l1[0]);
//            d_pix2ang(upix, disc_theta, disc_phi);

            disc_theta = HALFPI - b1[0];
            disc_phi = l1[0];
//            uphi = disc_phi - (d_const_GMaps.disc_size / cos(disc_theta));
            uphi = disc_phi - d_const_GMaps.disc_size;
            d_pix2ang(d_hpx_idx[0], hpx_zero_theta, hpx_zero_phi);
            double zero_angle = uphi * RAD2DEG;
//            printf("d_hpx_idx[0]=%f\n", zero_angle);
//            if (uphi < hpx_zero_phi){
//                uphi = hpx_zero_phi;
//            }
//            else
//                uphi = uphi;
//            printf("disc_theta=%f\n", disc_theta * RAD2DEG);
            uint64_t lpix = d_ang2pix(utheta, uphi); // disc size 范围首行的起始pixel
            if (disc_theta * RAD2DEG <= NORTH_B || disc_theta * RAD2DEG >= SOUTH_B){
                lpix = startpix;
            } else{
                lpix = lpix;
            }
//            uint64_t lpix = startpix;

            if (!(lpix >= startpix && lpix <= startpix + num_pix_in_ring)) {
                start_int = end_int;
                continue;
            }

//            uint64_t discs_size = d_const_GMaps.disc_size * RAD2DEG;
//            uint64_t center_pixel = d_ang2pix(utheta, disc_phi);

//            if ((l1[0] * RAD2DEG == 180) && (b1[0] * RAD2DEG == 0)){
//                printf("disc_phi=%f\n", disc_phi*RAD2DEG);
//                printf("uphi=%f\n", uphi*RAD2DEG);
//                printf("disc_theta=%f\n", disc_theta*RAD2DEG);
//                printf("startpix=%d\n", startpix);
//                printf("start_phi=%f\n", start_phi*RAD2DEG);
//                printf("lpix=%d\n", lpix);
//                printf("rpix=%d\n", d_ang2pix(utheta,disc_phi + d_const_GMaps.disc_size));
//                printf("center_pixel=%d\n", center_pixel);
//                printf("num_pixel_in_ring=%d\n", num_pix_in_ring);
//                printf("difference=%d\n", startpix - lpix);
//            }

//            uphi = disc_phi + (d_const_GMaps.disc_size / cos(disc_theta));
            uphi = disc_phi + d_const_GMaps.disc_size;
            uint64_t rpix = d_ang2pix(utheta, uphi);
            if (disc_theta * RAD2DEG <= NORTH_B || disc_theta * RAD2DEG >= SOUTH_B){
                rpix = startpix + num_pix_in_ring - 1;
            } else{
                rpix = rpix;
            }
//            uint64_t rpix = startpix + num_pix_in_ring - 1;
            if (!(rpix >= startpix && rpix < startpix + num_pix_in_ring)) {
                start_int = end_int;
                continue;
            }

            // find position of lpix
            uint32_t upix_idx = searchLastPosLessThan(d_hpx_idx, start_int - 1, end_int, lpix);
            ++upix_idx;
            if (upix_idx > end_int) {
                upix_idx = d_const_GMaps.data_shape;
            }

/*** 有效注释部分
//            printf("Lpix= %d\n", lpix);
//            printf("Startpix, Lpix, Rpix = %d, %d, %d\n", startpix, lpix, rpix);
//            printf("num_pixel_in_ring=%d\n", num_pix_in_ring);
//            printf("Lpix - StartPix=%d\n", lpix - startpix);
//            printf("Rpix - Lpix=%d\n", rpix - lpix);
//            printf("disc_size=%f\n", d_const_GMaps.disc_size * RAD2DEG);
//            printf("upix_idx = %d\n", upix_idx);
*/
//            printf("data_shape=%d", d_const_GMaps.data_shape);
//            if((l1[0] * RAD2DEG == 180) && (b1[0] * RAD2DEG == 0)){
//                printf("upix_idx=%d", upix_idx);
//                if (upix_idx < d_const_GMaps.data_shape){
//                    ++target_num;
//                    printf("target_num=%d\n", target_num);
//                }
//            }

            // Gridding
            while(upix_idx < d_const_GMaps.data_shape){
                double l2 = d_lons[upix_idx] * DEG2RAD;
                double b2 = d_lats[upix_idx] * DEG2RAD;
                upix = d_ang2pix(HALFPI - b2, l2);
                if (upix > rpix) {
                    break;
                }

                double in_weights = d_weights[upix_idx];
                double in_data = d_data[upix_idx];

                for (thread_id = left; thread_id <= right; ++thread_id) {
                    double sdist = true_angular_distance(l1[thread_id - left], b1[thread_id - left], l2, b2) * RAD2DEG;
                    double sbear = 0.;
                    if (d_const_GMaps.bearing_needed) {
                        sbear = great_circle_bearing(l1[thread_id - left], b1[thread_id - left], l2, b2);
                    }
//                    if ((l1[thread_id -left] * RAD2DEG == 180) && (b1[thread_id - left] * RAD2DEG== 0)){
//                        ++target_num;
//                        printf("target_num= %d", target_num);
//                        if (sdist < d_const_GMaps.sphere_radius){
//                            ++get_num;
//                            printf("get_num = %d", get_num);
//                        }
//                    }
                    if (sdist < d_const_GMaps.sphere_radius) {
                        double sweight = kernel_func_ptr(sdist, sbear);
                        double tweight = in_weights * sweight;
                        temp_data[thread_id - left] += in_data * tweight;
                        temp_weights[thread_id - left] += tweight;
                    }
                    d_datacube[thread_id] = temp_data[thread_id - left];
                    d_weightscube[thread_id] = temp_weights[thread_id - left];
                }
                ++upix_idx;
            }
            start_int = end_int;
            ++uring;
        }

//        for (thread_id = left; thread_id <= right; ++thread_id) {
//            d_datacube[thread_id] = temp_data[thread_id-left];
//            d_weightscube[thread_id] = temp_weights[thread_id-left];
//        }
    }
    __syncthreads();
}

/* Alloc data for GPU. */
void data_alloc(){
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t usedrings = h_Healpix.usedrings;

    HANDLE_ERROR(hipMalloc((void**)& d_lons, sizeof(double)*data_shape));
    HANDLE_ERROR(hipMalloc((void**)& d_lats, sizeof(double)*data_shape));
    HANDLE_ERROR(hipMalloc((void**)& d_data, sizeof(double)*data_shape));
    HANDLE_ERROR(hipMalloc((void**)& d_weights, sizeof(double)*data_shape));
    HANDLE_ERROR(hipMalloc((void**)& d_xwcs, sizeof(double)*num));
    HANDLE_ERROR(hipMalloc((void**)& d_ywcs, sizeof(double)*num));
    HANDLE_ERROR(hipMalloc((void**)& d_datacube, sizeof(double)*num));
    HANDLE_ERROR(hipMalloc((void**)& d_weightscube, sizeof(double)*num));
    HANDLE_ERROR(hipMalloc((void**)& d_hpx_idx, sizeof(uint64_t)*(data_shape+1)));
    HANDLE_ERROR(hipMalloc((void**)& d_start_ring, sizeof(uint32_t)*(usedrings+1)));
//    HANDLE_ERROR(hipBindTexture(NULL, tex_start_ring, d_start_ring, sizeof(uint32_t)*(usedrings+1)));
}

/* Send data from CPU to GPU. */
void data_h2d(){
    uint32_t data_shape = h_GMaps.data_shape;
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    uint32_t usedrings = h_Healpix.usedrings;

    // Copy constants memory
    HANDLE_ERROR(hipMemcpy(d_lons, h_lons, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_lats, h_lats, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_data, h_data, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_weights, h_weights, sizeof(double)*data_shape, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_xwcs, h_xwcs, sizeof(double)*num, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_ywcs,h_ywcs, sizeof(double)*num, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_datacube, h_datacube, sizeof(double)*num, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_weightscube, h_weightscube, sizeof(double)*num, hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_hpx_idx, h_hpx_idx, sizeof(uint64_t)*(data_shape+1), hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpy(d_start_ring, h_start_ring, sizeof(uint32_t)*(usedrings+1), hipMemcpyHostToDevice));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_kernel_params), h_kernel_params, sizeof(double)*3));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_zyx), h_zyx, sizeof(uint32_t)*3));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_Healpix), &h_Healpix, sizeof(Healpix)));
    HANDLE_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(d_const_GMaps), &h_GMaps, sizeof(GMaps)));
}

/* Send data from GPU to CPU. */
void data_d2h(){
    uint32_t num = h_zyx[0] * h_zyx[1] * h_zyx[2];
    HANDLE_ERROR(hipMemcpy(h_datacube, d_datacube, sizeof(double)*num, hipMemcpyDeviceToHost));
    HANDLE_ERROR(hipMemcpy(h_weightscube, d_weightscube, sizeof(double)*num, hipMemcpyDeviceToHost));
}

/* Release data. */
void data_free(){
    DEALLOC(h_lons);
    HANDLE_ERROR( hipFree(d_lons) );
    DEALLOC(h_lats);
    HANDLE_ERROR( hipFree(d_lats) );
    DEALLOC(h_data);
    HANDLE_ERROR( hipFree(d_data) );
    DEALLOC(h_weights);
    HANDLE_ERROR( hipFree(d_weights) );
    DEALLOC(h_xwcs);
    HANDLE_ERROR( hipFree(d_xwcs) );
    DEALLOC(h_ywcs);
    HANDLE_ERROR( hipFree(d_ywcs) );
    DEALLOC(h_datacube);
    HANDLE_ERROR( hipFree(d_datacube) );
    DEALLOC(h_weightscube);
    HANDLE_ERROR( hipFree(d_weightscube) );
    DEALLOC(h_hpx_idx);
    HANDLE_ERROR( hipFree(d_hpx_idx) );
    DEALLOC(h_start_ring);
//    HANDLE_ERROR( hipUnbindTexture(tex_start_ring) );

    HANDLE_ERROR( hipFree(d_start_ring) );
    DEALLOC(h_header);
    DEALLOC(h_zyx);
    DEALLOC(h_kernel_params);
}

/* Gridding process. */
void solve_gridding(const char *infile, const char *tarfile, const char *outfile, const char *sortfile, const int& param, const int &bDim) {
    double iTime1 = cpuSecond();
    // Read input points.
    read_input_map(infile);

    // Read output map.
    read_output_map(tarfile);

    // Set wcs for output pixels.
    set_WCS();

    // Initialize output spectrals and weights.
    init_output();

//    iTime2 = cpuSecond();
    // Block Indirect Sort input points by their healpix indexes.
//    if (param == THRUST) {
//        init_input_with_thrust(param);
//    } else {
//        init_input_with_cpu(param);
//    }

    init_input_with_cpu(param);
    double iTime3 = cpuSecond();
    // Alloc data for GPU.
    data_alloc();

    double iTime4 = cpuSecond();
    // Send data from CPU to GPU.
    data_h2d();
    printf("h_zyx[1]=%d, h_zyx[2]=%d, ", h_zyx[1], h_zyx[2]);

    // Set block and thread.
    dim3 block(bDim);
    dim3 grid((h_GMaps.block_warp_num * h_zyx[1] - 1) / (block.x / 32) + 1);
    printf("grid.x=%d, block.x=%d, ", grid.x, block.x);

    // Get start time.
    hipEvent_t start, stop;
    HANDLE_ERROR(hipEventCreate(&start));
    HANDLE_ERROR(hipEventCreate(&stop));
    HANDLE_ERROR(hipEventRecord(start, 0));

    // Call device kernel.
    hipLaunchKernelGGL(hcgrid, dim3(grid), dim3(block ), 0, 0, d_lons, d_lats, d_data, d_weights, d_xwcs, d_ywcs, d_datacube, d_weightscube, d_start_ring, d_hpx_idx);

    // Get stop time.
    HANDLE_ERROR(hipEventRecord(stop, 0));
    HANDLE_ERROR(hipEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(hipEventElapsedTime(&elapsedTime, start, stop));
    printf("kernel elapsed time=%f, ", elapsedTime);

    // Send data from GPU to CPU
    data_d2h();

    // Write output FITS file
    write_output_map(outfile);

    // Write sorted input FITS file
    if (sortfile) {
        write_ordered_map(infile, sortfile);
    }

    // Release data
    data_free();
    HANDLE_ERROR( hipEventDestroy(start) );
    HANDLE_ERROR( hipEventDestroy(stop) );
    HANDLE_ERROR( hipDeviceReset() );

    double iTime5 = cpuSecond();
    double iElaps = (iTime5 - iTime1) * 1000.;
    printf("solving_gridding time=%f\n", iElaps);
}

