#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "uwnet.h"
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"

int tests_total = 0;
int tests_fail = 0;


int within_eps(float a, float b){
    return a-EPS<b && b<a+EPS;
}

int same_matrix(matrix a, matrix b)
{
    int i;
    if(a.rows != b.rows || a.cols != b.cols) {
        //printf ("first matrix: %dx%d, second matrix:%dx%d\n", a.rows, a.cols, b.rows, b.cols);
        return 0;
    }
    for(i = 0; i < a.rows*a.cols; ++i){
        if(!within_eps(a.data[i], b.data[i])) {
            //printf("differs at %d, %f vs %f\n", i, a.data[i], b.data[i]);
            return 0;
        }
    }
    return 1;
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void test_copy_matrix()
{
    matrix a = random_matrix(32, 64, 10);
    matrix c = copy_matrix(a);
    TEST(same_matrix(a,c));
    free_matrix(a);
    free_matrix(c);
}

void test_transpose_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix at = load_matrix("data/test/at.matrix");
    matrix atest = transpose_matrix(a);
    matrix aorig = transpose_matrix(atest);
    TEST(same_matrix(at, atest) && same_matrix(a, aorig));
    free_matrix(a);
    free_matrix(at);
    free_matrix(atest);
    free_matrix(aorig);
}

void test_axpy_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix y1 = load_matrix("data/test/y1.matrix");
    axpy_matrix(2, a, y);
    TEST(same_matrix(y, y1));
    free_matrix(a);
    free_matrix(y);
    free_matrix(y1);
}

void test_matmul()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix c = load_matrix("data/test/c.matrix");
    matrix mul = matmul(a, b);
    TEST(same_matrix(c, mul));
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(mul);
}

void test_activate_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_alog = load_matrix("data/test/alog.matrix");
    matrix truth_arelu = load_matrix("data/test/arelu.matrix");
    matrix truth_alrelu = load_matrix("data/test/alrelu.matrix");
    matrix truth_asoft = load_matrix("data/test/asoft.matrix");
    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    TEST(same_matrix(truth_alog, alog));
    TEST(same_matrix(truth_arelu, arelu));
    TEST(same_matrix(truth_alrelu, alrelu));
    TEST(same_matrix(truth_asoft, asoft));
    free_matrix(a);
    free_matrix(alog);
    free_matrix(arelu);
    free_matrix(alrelu);
    free_matrix(asoft);
    free_matrix(truth_alog);
    free_matrix(truth_arelu);
    free_matrix(truth_alrelu);
    free_matrix(truth_asoft);
}

void test_gradient_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix truth_glog = load_matrix("data/test/glog.matrix");
    matrix truth_grelu = load_matrix("data/test/grelu.matrix");
    matrix truth_glrelu = load_matrix("data/test/glrelu.matrix");
    matrix truth_gsoft = load_matrix("data/test/gsoft.matrix");
    matrix glog = copy_matrix(a);
    matrix grelu = copy_matrix(a);
    matrix glrelu = copy_matrix(a);
    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    gradient_matrix(y, RELU, grelu);
    gradient_matrix(y, LRELU, glrelu);
    gradient_matrix(y, SOFTMAX, gsoft);
    TEST(same_matrix(truth_glog, glog));
    TEST(same_matrix(truth_grelu, grelu));
    TEST(same_matrix(truth_glrelu, glrelu));
    TEST(same_matrix(truth_gsoft, gsoft));
    free_matrix(a);
    free_matrix(glog);
    free_matrix(grelu);
    free_matrix(glrelu);
    free_matrix(gsoft);
    free_matrix(truth_glog);
    free_matrix(truth_grelu);
    free_matrix(truth_glrelu);
    free_matrix(truth_gsoft);
}

void test_connected_layer()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix dw = load_matrix("data/test/dw.matrix");
    matrix db = load_matrix("data/test/db.matrix");
    matrix delta = load_matrix("data/test/delta.matrix");
    matrix prev_delta = load_matrix("data/test/prev_delta.matrix");
    matrix truth_prev_delta = load_matrix("data/test/truth_prev_delta.matrix");
    matrix truth_dw = load_matrix("data/test/truth_dw.matrix");
    matrix truth_db = load_matrix("data/test/truth_db.matrix");
    matrix updated_dw = load_matrix("data/test/updated_dw.matrix");
    matrix updated_db = load_matrix("data/test/updated_db.matrix");
    matrix updated_w = load_matrix("data/test/updated_w.matrix");
    matrix updated_b = load_matrix("data/test/updated_b.matrix");

    matrix bias = load_matrix("data/test/bias.matrix");
    matrix truth_out = load_matrix("data/test/out.matrix");
    layer l = make_connected_layer(64, 16, LRELU);
    l.w = b;
    l.b = bias;
    l.dw = dw;
    l.db = db;
    matrix out = l.forward(l, a);
    TEST(same_matrix(truth_out, out));


    l.delta[0] = delta;
    l.backward(l, prev_delta);
    TEST(same_matrix(truth_prev_delta, prev_delta));
    TEST(same_matrix(truth_dw, l.dw));
    TEST(same_matrix(truth_db, l.db));

    l.update(l, .01, .9, .01);
    TEST(same_matrix(updated_dw, l.dw));
    TEST(same_matrix(updated_db, l.db));
    TEST(same_matrix(updated_w, l.w));
    TEST(same_matrix(updated_b, l.b));

    free_matrix(a);
    free_matrix(b);
    free_matrix(bias);
    free_matrix(out);
    free_matrix(truth_out);
}

void test_matrix_mean()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_mu_a = load_matrix("data/test/mu_a.matrix");
    matrix truth_mu_a_s = load_matrix("data/test/mu_a_s.matrix");

    matrix mu_a   = mean(a, 1);
    matrix mu_a_s = mean(a, 8);

    TEST(same_matrix(truth_mu_a, mu_a));
    TEST(same_matrix(truth_mu_a_s, mu_a_s));

    free_matrix(a);
    free_matrix(mu_a);
    free_matrix(mu_a_s);
    free_matrix(truth_mu_a);
    free_matrix(truth_mu_a_s);
}

void test_matrix_variance()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");

    matrix sig_a    =  variance(a, mu_a, 1);
    matrix sig_a_s  =  variance(a, mu_a_s, 8);

    matrix truth_sig_a = load_matrix("data/test/sig_a.matrix");
    matrix truth_sig_a_s = load_matrix("data/test/sig_a_s.matrix");

    TEST(same_matrix(truth_sig_a, sig_a));
    TEST(same_matrix(truth_sig_a_s, sig_a_s));

    free_matrix(a);
    free_matrix(mu_a);
    free_matrix(mu_a_s);
    free_matrix(sig_a);
    free_matrix(sig_a_s);
    free_matrix(truth_sig_a);
    free_matrix(truth_sig_a_s);
}

void test_matrix_normalize()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix truth_norm_a   = load_matrix("data/test/norm_a.matrix");
    matrix truth_norm_a_s = load_matrix("data/test/norm_a_s.matrix");

    matrix norm_a = normalize(a, mu_a, sig_a, 1);
    matrix norm_a_s = normalize(a, mu_a_s, sig_a_s, 8);

    TEST(same_matrix(truth_norm_a,   norm_a));
    TEST(same_matrix(truth_norm_a_s, norm_a_s));

    free_matrix(a);
    free_matrix(mu_a);
    free_matrix(mu_a_s);
    free_matrix(sig_a);
    free_matrix(sig_a_s);
    free_matrix(norm_a);
    free_matrix(norm_a_s);
    free_matrix(truth_norm_a);
    free_matrix(truth_norm_a_s);
}

void test_matrix_delta_mean()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix d        = load_matrix("data/test/y.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix truth_dm       = load_matrix("data/test/dm.matrix");
    matrix truth_dm_s     = load_matrix("data/test/dm_s.matrix");

    matrix dm = delta_mean(d, sig_a, 1);
    matrix dm_s = delta_mean(d, sig_a_s, 8);

    TEST(same_matrix(truth_dm,   dm));
    TEST(same_matrix(truth_dm_s, dm_s));

    free_matrix(a);
    free_matrix(mu_a);
    free_matrix(mu_a_s);
    free_matrix(sig_a);
    free_matrix(sig_a_s);
}

void test_matrix_delta_variance()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix d        = load_matrix("data/test/y.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix truth_dv       = load_matrix("data/test/dv.matrix");
    matrix truth_dv_s     = load_matrix("data/test/dv_s.matrix");

    matrix dv = delta_variance(d, a, mu_a, sig_a, 1);
    matrix dv_s = delta_variance(d, a, mu_a_s, sig_a_s, 8);

    TEST(same_matrix(truth_dv,   dv));
    TEST(same_matrix(truth_dv_s, dv_s));

    free_matrix(a);
    free_matrix(mu_a);
    free_matrix(mu_a_s);
    free_matrix(sig_a);
    free_matrix(sig_a_s);
}

void test_matrix_delta_normalize()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix d        = load_matrix("data/test/y.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix dm       = load_matrix("data/test/dm.matrix");
    matrix dm_s     = load_matrix("data/test/dm_s.matrix");
    matrix dv       = load_matrix("data/test/dv.matrix");
    matrix dv_s     = load_matrix("data/test/dv_s.matrix");
    matrix truth_dbn      = load_matrix("data/test/dbn.matrix");
    matrix truth_dbn_s    = load_matrix("data/test/dbn_s.matrix");

    matrix dbn = delta_batch_norm(d, dm, dv, mu_a, sig_a, a, 1);
    matrix dbn_s = delta_batch_norm(d, dm_s, dv_s, mu_a_s, sig_a_s, a, 8);

    TEST(same_matrix(truth_dbn,   dbn));
    TEST(same_matrix(truth_dbn_s, dbn_s));

    free_matrix(a);
    free_matrix(mu_a);
    free_matrix(mu_a_s);
    free_matrix(sig_a);
    free_matrix(sig_a_s);
}

void test_im2col()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix col = im2col(im, 3, 2);
    matrix truth_col = load_matrix("data/test/im2col.matrix");
    matrix col2 = im2col(im, 2, 2);
    matrix truth_col2 = load_matrix("data/test/im2col2.matrix");
    TEST(same_matrix(truth_col,   col));
    TEST(same_matrix(truth_col2,  col2));
    free_matrix(col);
    free_matrix(col2);
    free_matrix(truth_col);
    free_matrix(truth_col2);
    free_image(im);
}

void test_col2im()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix dcol = load_matrix("data/test/dcol.matrix");
    matrix dcol2 = load_matrix("data/test/dcol2.matrix");
    image col2im_res = make_image(im.w, im.h, im.c);
    image col2im_res2 = make_image(im.w, im.h, im.c);
    col2im(dcol, 3, 2, col2im_res);
    col2im(dcol2, 2, 2, col2im_res2);
    matrix col2mat2 = {0};
    col2mat2.rows = col2im_res2.c;
    col2mat2.cols = col2im_res2.w*col2im_res2.h;
    col2mat2.data = col2im_res2.data;

    matrix col2mat = {0};
    col2mat.rows = col2im_res.c;
    col2mat.cols = col2im_res.w*col2im_res.h;
    col2mat.data = col2im_res.data;
    matrix truth_col2mat = load_matrix("data/test/col2mat.matrix");
    matrix truth_col2mat2 = load_matrix("data/test/col2mat2.matrix");
    TEST(same_matrix(truth_col2mat, col2mat));
    TEST(same_matrix(truth_col2mat2, col2mat2));
    free_matrix(dcol);
    free_matrix(col2mat);
    free_matrix(truth_col2mat);
    free_matrix(dcol2);
    free_matrix(col2mat2);
    free_matrix(truth_col2mat2);
    free_image(im);
}

void test_maxpool_layer_forward()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;
    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;
    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    matrix max_out = max_l.forward(max_l, im_mat);
    matrix truth_max_out = load_matrix("data/test/max_out.matrix");
    TEST(same_matrix(truth_max_out, max_out));
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);
    matrix max_out3 = max_l3.forward(max_l3, im_mat3);
    matrix truth_max_out3 = load_matrix("data/test/max_out3.matrix");
    TEST(same_matrix(truth_max_out3, max_out3));
    free_matrix(max_out);
    free_matrix(truth_max_out);
    free_matrix(max_out3);
    free_matrix(truth_max_out3);
    free_image(im);
}

void test_maxpool_layer_backward()
{
    matrix truth_max_out = load_matrix("data/test/max_out.matrix");
    matrix truth_max_out3 = load_matrix("data/test/max_out3.matrix");
    image im = load_image("data/test/dog.jpg"); 
    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;
    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    max_l.in[0] = im_mat;
    max_l.out[0] = truth_max_out;

    matrix max_delta = load_matrix("data/test/max_delta.matrix");
    matrix prev_max_delta = make_matrix(im_mat.rows, im_mat.cols);

    *max_l.delta = max_delta;
    max_l.backward(max_l, prev_max_delta);
    matrix truth_prev_max_delta = load_matrix("data/test/prev_max_delta.matrix");
    TEST(same_matrix(truth_prev_max_delta, prev_max_delta));

    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);
    max_l3.in[0] = im_mat3;
    max_l3.out[0] = truth_max_out3;

    matrix max_delta3 = load_matrix("data/test/max_delta3.matrix");
    matrix prev_max_delta3 = make_matrix(im_mat3.rows, im_mat3.cols);

    *max_l3.delta = max_delta3;
    max_l3.backward(max_l3, prev_max_delta3);
    matrix truth_prev_max_delta3 = load_matrix("data/test/prev_max_delta3.matrix");
    TEST(same_matrix(truth_prev_max_delta3, prev_max_delta3));
    free_matrix(max_delta);
    free_matrix(prev_max_delta);
    free_matrix(truth_prev_max_delta);
    free_matrix(max_delta3);
    free_matrix(prev_max_delta3);
    free_matrix(truth_prev_max_delta3);
    free_image(im);
}

void make_matrix_test()
{
    srand(1);
    matrix a = random_matrix(32, 64, 10);
    matrix b = random_matrix(64, 16, 10);
    matrix at = transpose_matrix(a);
    matrix c = matmul(a, b);
    matrix y = random_matrix(32, 64, 10);
    matrix bias = random_matrix(1, 16, 10);
    matrix dw = random_matrix(64, 16, 10);
    matrix db = random_matrix(1, 16, 10);
    matrix delta = random_matrix(32, 16, 10);
    matrix prev_delta = random_matrix(32, 64, 10);
    matrix y1 = copy_matrix(y);
    axpy_matrix(2, a, y1);
    save_matrix(a, "data/test/a.matrix");
    save_matrix(b, "data/test/b.matrix");
    save_matrix(bias, "data/test/bias.matrix");
    save_matrix(dw, "data/test/dw.matrix");
    save_matrix(db, "data/test/db.matrix");
    save_matrix(at, "data/test/at.matrix");
    save_matrix(delta, "data/test/delta.matrix");
    save_matrix(prev_delta, "data/test/prev_delta.matrix");
    save_matrix(c, "data/test/c.matrix");
    save_matrix(y, "data/test/y.matrix");
    save_matrix(y1, "data/test/y1.matrix");

    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    save_matrix(alog, "data/test/alog.matrix");

    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    save_matrix(arelu, "data/test/arelu.matrix");

    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    save_matrix(alrelu, "data/test/alrelu.matrix");

    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    save_matrix(asoft, "data/test/asoft.matrix");



    matrix glog = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    save_matrix(glog, "data/test/glog.matrix");

    matrix grelu = copy_matrix(a);
    gradient_matrix(y, RELU, grelu);
    save_matrix(grelu, "data/test/grelu.matrix");

    matrix glrelu = copy_matrix(a);
    gradient_matrix(y, LRELU, glrelu);
    save_matrix(glrelu, "data/test/glrelu.matrix");

    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, SOFTMAX, gsoft);
    save_matrix(gsoft, "data/test/gsoft.matrix");


    layer l = make_connected_layer(64, 16, LRELU);
    l.w = b;
    l.b = bias;
    l.dw = dw;
    l.db = db;

    matrix out = l.forward(l, a);
    save_matrix(out, "data/test/out.matrix");

    l.delta[0] = delta;
    l.backward(l, prev_delta);
    save_matrix(prev_delta, "data/test/truth_prev_delta.matrix");
    save_matrix(l.dw, "data/test/truth_dw.matrix");
    save_matrix(l.db, "data/test/truth_db.matrix");

    l.update(l, .01, .9, .01);
    save_matrix(l.dw, "data/test/updated_dw.matrix");
    save_matrix(l.db, "data/test/updated_db.matrix");
    save_matrix(l.w, "data/test/updated_w.matrix");
    save_matrix(l.b, "data/test/updated_b.matrix");

    image im = load_image("data/test/dog.jpg"); 
    matrix col = im2col(im, 3, 2);
    matrix col2 = im2col(im, 2, 2);
    save_matrix(col, "data/test/im2col.matrix");
    save_matrix(col2, "data/test/im2col2.matrix");

    matrix dcol = random_matrix(col.rows, col.cols, 10);
    matrix dcol2 = random_matrix(col2.rows, col2.cols, 10);
    image col2im_res = make_image(im.w, im.h, im.c);
    image col2im_res2 = make_image(im.w, im.h, im.c);
    col2im(dcol, 3, 2, col2im_res);
    col2im(dcol2, 2, 2, col2im_res2);
    save_matrix(dcol, "data/test/dcol.matrix");
    save_matrix(dcol2, "data/test/dcol2.matrix");
    matrix col2mat = {0};
    col2mat.rows = col2im_res.c;
    col2mat.cols = col2im_res.w*col2im_res.h;
    col2mat.data = col2im_res.data;
    save_matrix(col2mat, "data/test/col2mat.matrix");
    matrix col2mat2 = {0};
    col2mat2.rows = col2im_res2.c;
    col2mat2.cols = col2im_res2.w*col2im_res2.h;
    col2mat2.data = col2im_res2.data;
    save_matrix(col2mat2, "data/test/col2mat2.matrix");


    // Maxpool Layer Tests

    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;
    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    matrix max_out = max_l.forward(max_l, im_mat);
    save_matrix(max_out, "data/test/max_out.matrix");

    matrix max_delta = random_matrix(max_out.rows, max_out.cols, 10);
    save_matrix(max_delta, "data/test/max_delta.matrix");

    matrix prev_max_delta = make_matrix(im_mat.rows, im_mat.cols);
    *max_l.delta = max_delta;
    max_l.backward(max_l, prev_max_delta);
    save_matrix(prev_max_delta, "data/test/prev_max_delta.matrix");

    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);
    matrix max_out3 = max_l.forward(max_l3, im_mat3);
    save_matrix(max_out3, "data/test/max_out3.matrix");

    matrix max_delta3 = random_matrix(max_out3.rows, max_out3.cols, 10);
    save_matrix(max_delta3, "data/test/max_delta3.matrix");

    matrix prev_max_delta3 = make_matrix(im_mat3.rows, im_mat3.cols);
    *max_l3.delta = max_delta3;
    max_l.backward(max_l3, prev_max_delta3);
    save_matrix(prev_max_delta3, "data/test/prev_max_delta3.matrix");


    // Batchnorm Tests

    matrix mu_a   = mean(a, 1);
    matrix mu_a_s = mean(a, 8);

    matrix sig_a   =  variance(a, mu_a, 1);
    matrix sig_a_s =  variance(a, mu_a_s, 8);

    matrix norm_a = normalize(a, mu_a, sig_a, 1);
    matrix norm_a_s = normalize(a, mu_a_s, sig_a_s, 8);

    save_matrix(mu_a, "data/test/mu_a.matrix");
    save_matrix(mu_a_s, "data/test/mu_a_s.matrix");
    save_matrix(sig_a, "data/test/sig_a.matrix");
    save_matrix(sig_a_s, "data/test/sig_a_s.matrix");
    save_matrix(norm_a, "data/test/norm_a.matrix");
    save_matrix(norm_a_s, "data/test/norm_a_s.matrix");

    matrix dm = delta_mean(y, sig_a, 1);
    matrix dm_s = delta_mean(y, sig_a_s, 8);

    save_matrix(dm, "data/test/dm.matrix");
    save_matrix(dm_s, "data/test/dm_s.matrix");

    matrix dv = delta_variance(y, a, mu_a, sig_a, 1);
    matrix dv_s = delta_variance(y, a, mu_a_s, sig_a_s, 8);

    save_matrix(dv, "data/test/dv.matrix");
    save_matrix(dv_s, "data/test/dv_s.matrix");

    matrix dbn = delta_batch_norm(y, dm, dv, mu_a, sig_a, a, 1);
    matrix dbn_s = delta_batch_norm(y, dm_s, dv_s, mu_a_s, sig_a_s, a, 8);
    save_matrix(dbn, "data/test/dbn.matrix");
    save_matrix(dbn_s, "data/test/dbn_s.matrix");
}

void test_matrix_speed()
{
    int i;
    int n = 128;
    matrix a = random_matrix(512, 512, 1);
    matrix b = random_matrix(512, 512, 1);
    double start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix d = matmul(a,b);
        free_matrix(d);
    }
    printf("Matmul elapsed %lf sec\n", what_time_is_it_now() - start);
    start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix at = transpose_matrix(a);
        free_matrix(at);
    }
    printf("Transpose elapsed %lf sec\n", what_time_is_it_now() - start);
}

void run_tests()
{
    //make_matrix_test();
    test_copy_matrix();
    test_axpy_matrix();
    test_transpose_matrix();
    test_matmul();
    test_activate_matrix();
    test_gradient_matrix();
    test_connected_layer();
    test_im2col();
    test_col2im();
    test_maxpool_layer_forward();
    test_maxpool_layer_backward();
    test_matrix_mean();
    test_matrix_variance();
    test_matrix_normalize();
    test_matrix_delta_mean();
    test_matrix_delta_variance();
    test_matrix_delta_normalize();
    //test_matrix_speed();
    //printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

