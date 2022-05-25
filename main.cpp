#include <iostream>
#include <opencv2/opencv.hpp>    
#include <opencv2/imgproc/imgproc.hpp>   


void show(std::string img_name, cv::Mat img) {
    cv::namedWindow(img_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(img_name, 1000, 1000);
    cv::imshow(img_name, img);
}

int dis(int cx, int cy, int tx, int ty) {
    return pow(pow(cx - tx, 2) + pow(cy - ty, 2), 0.5);

}


int find_contour(cv::Mat img) {//�ҳ�����Ʊ������

    cv::Mat img_rect;   // ���ƾ�ȷ����
    bilateralFilter(img, img_rect, 0, 100, 10);//˫��ģ��
    //GaussianBlur(img, img_rect, Size(5, 5), 15);//��˹ģ��
    //img.copyTo(img_rect);
    img = img_rect;
    //show("img", img);
 
    cv::Mat img_gray, img_binary;
    std::vector<std::vector<cv::Point>> contours;   //���������ļ���
    std::vector<std::vector<cv::Point>> contours_select;   //ɸѡ�������ļ���

    //ת��Ϊ�Ҷ�ͼ
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    //cv::cvtColor(img, img_gray, cv::COLOR_BGR2HSV);

    // ��ֵ��
    adaptiveThreshold(~img_gray, img_binary, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -10);

    // ��Ե���
    cv::findContours(img_binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    //cv::findContours(img_gray, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // ɸѡ����������������
    int maxrec_index = 0;   //��������Ӿ��ε�index
    int maxarea = -1;
    for (int i = 0; i < contours.size(); ++i) {
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        double w = rect.size.width;
        double h = rect.size.height;
        if (w * h > maxarea) {
            maxarea = w * h;
            maxrec_index = i;
        }
    }

    contours_select.push_back(contours[maxrec_index]);
        cv::Mat img_maxrec;
        img.copyTo(img_maxrec);   //���
        cv::drawContours(img_maxrec, contours_select, -1,
                         cv::Scalar(0, 255, 0), 2);    //  ������С��������

        //������С��Ӿ���
        cv::Point2f vertices[4];
        cv::RotatedRect rect = minAreaRect(contours[maxrec_index]);
        rect.points(vertices);
        std::cout << rect.center;
        cv::Mat img_minrec;
        img.copyTo(img_minrec);   //���
        for (int i = 0; i < 4; ++i) {
            line(img_minrec, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        }
        cv::Rect brect = rect.boundingRect();
        std::cout << brect;
        cv::rectangle(img, brect, cv::Scalar(255, 0, 0), 2);

    int sumx = 0;
    int sumy = 0;
    for (int i = 0; i < contours_select[0].size(); ++i) {
        sumx += contours_select[0][i].x;
        sumy += contours_select[0][i].y;
    }
    int centerx = sumx / contours_select[0].size();
    int centery = sumy / contours_select[0].size();

        std::vector<std::vector<cv::Point>> approx(contours.size());//���ֱ�߲�����ֱ��
        for (int i = 0; i < contours.size(); i++) {
            approxPolyDP(cv::Mat(contours[i]), approx[i], 15, true);
    
            drawContours(img, approx, i, cv::Scalar(0, 255, 255), 2, 8);  //����
        }


        // �ҳ���ȷ����Ʊ����
    std::vector<cv::Point> cor(4);
    int maxdis[4] = { 0,0,0,0 };
    for (int i = 0; i < contours_select[0].size(); i++) {

        if (contours_select[0][i].x < centerx && contours_select[0][i].y < centery) { // left top point
            if (dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y) > maxdis[0]) {
                maxdis[0] = dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y);
                cor[0].x = contours_select[0][i].x;
                cor[0].y = contours_select[0][i].y;
            }
        }
        else if (contours_select[0][i].x > centerx && contours_select[0][i].y < centery) {  //  right top point
            if (dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y) > maxdis[1]) {
                maxdis[1] = dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y);
                cor[1].x = contours_select[0][i].x;
                cor[1].y = contours_select[0][i].y;
            }
        }
        else if (contours_select[0][i].x > centerx && contours_select[0][i].y > centery) {  //  right down point
            if (dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y) > maxdis[2]) {
                maxdis[2] = dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y);
                cor[2].x = contours_select[0][i].x;
                cor[2].y = contours_select[0][i].y;
            }
        }
        else if (contours_select[0][i].x < centerx && contours_select[0][i].y > centery) {  //  left down point
            if (dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y) > maxdis[3]) {
                maxdis[3] = dis(centerx, centery, contours_select[0][i].x, contours_select[0][i].y);
                cor[3].x = contours_select[0][i].x;
                cor[3].y = contours_select[0][i].y;
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        line(img_rect, cor[i], cor[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }

    show("img", img);
    show("img_binary", img_binary);
        show("img_maxrec", img_maxrec);
        show("img_minrec", img_minrec);
    show("img_rect", img_rect);


    cv::waitKey(0);
    return 1;
}

int main() {
    cv::Mat img = cv::imread("E:/OpenCV_image/7.png");
    find_contour(img);
    return 1;
}


//#include<opencv2/opencv.hpp>
//#include<iostream>
//#include<opencv2/highgui/highgui_c.h>
//#include<vector>
//
//using namespace cv;
//using namespace std;
//
//
//int main(int argv, char** argc)
//{
//	Mat src;
//	src = imread("E://OpenCV_image//7.png");
//	if (!src.data)
//	{
//		cout << "Could not loaded image..." << endl;
//		return -1;
//	}
//
//	// convert binary image
//	Mat grayImg, binImg;
//	cvtColor(src, grayImg, COLOR_BGR2GRAY);
//	threshold(grayImg, binImg, 100, 255, THRESH_BINARY | THRESH_OTSU);        //ͼ���ֵ��
//
//	// find contours
//	vector<vector<cv::Point>> contours;
//	vector<cv::Vec4i> hierarchy;
//	findContours(binImg, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE); //��ȡ����
//
//	// draw find result                                                 
//	for (size_t i = 0; i < contours.size(); i++)
//	{
//		//���Ƶ�i������
//		drawContours(src, contours, (int)i, Scalar(0, 0, 255), 2, 8, hierarchy, 0);
//	}
//
//	// show find result
//	namedWindow("Find Result", WINDOW_AUTOSIZE);
//	imshow("Find Result", src);
//	waitKey(0);
//	return 0;
//}
