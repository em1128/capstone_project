#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
#define MOTION_NUM 19
static float Pi = 3.141592;
static int joint_num = 28;

// 3D 포인트 구조체
struct Point3D {
    float x, y, z;
    Point3D(vector<double> v) {
        x = v[0];
        y = v[1];
        z = v[2];
    }
    Point3D(float x, float y, float z) : x(x), y(y), z(z) {}
    Point3D() : x(x), y(y), z(z) {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }
    
    void mul(double a) {
        x *= a;
        y *= a;
        z *= a;
    }
    void div(double a) {
        
        if (a < 10e-9) {
            x = 0; y = 0; z = 0;
        }
        else {
            x /= a;
            y /= a;
            z /= a;
        }
    }
    void addv(Point3D p) {
        x += p.x;
        y += p.y;
        z += p.z;
    }
    void minv(Point3D p) {
        x -= p.x;
        y -= p.y;
        z -= p.z;
    }
    // return이 Point3D
    Point3D vmin(Point3D p) {
        p.x = x-p.x;
        p.y = y - p.y;
        p.z = z - p.z;
        return p;
    }
    Point3D vmul(double a) {
        return Point3D(x * a, y * a, z * a);
    }
    //
    void print() {
        cout << "x : " << x << " y : " << y << " z : " << z << '\n';
    }
};

static vector<int> joints_hierarchy;
// 로드리게스 회전 함수
Point3D RodriguesRotation(Point3D inputPoint, Point3D axis, float angle) {

    // 로드리게스 회전 공식 적용
    float cosTheta = cos(angle);
    float sinTheta = sin(angle);
    float dotProduct = inputPoint.x * axis.x + inputPoint.y * axis.y + inputPoint.z * axis.z;

    Point3D crossProduct;
    crossProduct.x = inputPoint.y * axis.z - inputPoint.z * axis.y;
    crossProduct.y = inputPoint.z * axis.x - inputPoint.x * axis.z;
    crossProduct.z = inputPoint.x * axis.y - inputPoint.y * axis.x;

    Point3D result;
    result.x = inputPoint.x * cosTheta + crossProduct.x * sinTheta + axis.x * dotProduct * (1 - cosTheta);
    result.y = inputPoint.y * cosTheta + crossProduct.y * sinTheta + axis.y * dotProduct * (1 - cosTheta);
    result.z = inputPoint.z * cosTheta + crossProduct.z * sinTheta + axis.z * dotProduct * (1 - cosTheta);
    // normalize
    float div = sqrt(pow(result.x, 2) + pow(result.y, 2) + pow(result.z, 2));
    result.div(div);
    
    return result;
}
Point3D RodriguesRotation2(Point3D inputPoint, Point3D axis, float angle) {
    Point3D vector = axis.vmul(angle);
    cv::Mat input = (cv::Mat_<double>(3, 1) << inputPoint.x, inputPoint.y, inputPoint.z);
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << vector.x, vector.y, vector.z);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    input = R * input;
    Point3D result(input.at<double>(0, 0), input.at<double>(1, 0), input.at<double>(2, 0));
    // normalize
    float div = sqrt(pow(result.x, 2) + pow(result.y, 2) + pow(result.z, 2));
    result.div(div);

    return result;
}
vector<Point3D> Local2global(vector<Point3D> global) {
    vector<Point3D> local;
    local.assign(global.size(), Point3D());

    
    for (int j = 0; j < 28; ++j) {
        
        if (j == 9 || j == 16 || j == 17 || j == 10) { continue; }
        cv::Mat rod_g = (cv::Mat_<double>(3, 1) << global[j].x, global[j].y, global[j].z);
        cv::Mat R;
        int parent_j = joints_hierarchy[j];
        while (parent_j != -1) {
            cv::Mat rvec = (cv::Mat_<double>(3, 1) << global[parent_j].x, global[parent_j].y, global[parent_j].z);
            cv::Rodrigues(rvec, R);
            rod_g = R * rod_g;
            parent_j = joints_hierarchy[parent_j];
        }
        local[j] = Point3D(rod_g.at<double>(0, 0), rod_g.at<double>(1, 0), rod_g.at<double>(2, 0));
        
    }
    return local;
}
void drawDot(Point3D p, int numPoints, float radius, vector<Point3D>& pointCloud) {
    Point3D dot_point;
    // Create a sphere point cloud and write it to the file
    for (int i = 0; i < numPoints; ++i) {
        double theta = 2.0 * Pi * i / numPoints; // Azimuthal angle
        for (int j = 0; j < numPoints * (i + 1); ++j) {
            double phi = Pi * j / numPoints; // Polar angle
            dot_point.x = radius * sin(phi) * cos(theta);
            dot_point.y = radius * sin(phi) * sin(theta);
            dot_point.z = radius * cos(phi);
            dot_point.addv(p);
            pointCloud.push_back(dot_point);
        }
    }
}
void drawLine(Point3D p1, Point3D p2, int numPoints, vector<Point3D>& pointCloud) {
    Point3D line_point;
    Point3D direction = p2.vmin(p1); //자식에서 부모로(p1:자식, p2:부모)
    
    for (int i = 0; i < numPoints; ++i) {

        double t = static_cast<double>(i+1) / numPoints;
        line_point = direction.vmul(t);
        line_point.addv(p1);

        pointCloud.push_back(line_point);
    }
}
void visualizePoints(vector<Point3D> points, vector<Point3D>& pointCloud) {
    float radius = 1;
    int numPoints1 = 8;
    int numPoints2 = 20;

    for (int j = 0; j < joint_num;++j) {
        drawDot(points[j], numPoints1, radius, pointCloud);
        if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
        drawLine(points[j],points[joints_hierarchy[j]],numPoints2, pointCloud);
    }
}
vector<vector<vector<vector<double>>>> load_ans_vector();
vector<double> normalize_vectors(vector<Point3D>& v);
void normalize_point(Point3D& v);
void  static Hierarchy_set();

int main() {
    
    Hierarchy_set();
    // [motion_index][max_frames][joints][xyz]
    vector<vector<vector<vector<double>>>> total_ans_vector;
    total_ans_vector = load_ans_vector();

    Point3D zeros;
    vector<Point3D> rvec;
    rvec.assign(joint_num, zeros);
    
    for (int joint = 0; joint < joint_num; ++joint) {
        rvec[joint] = total_ans_vector[1][0][joint];
    }
    vector<Point3D> rotationAxes;
    vector<double> rotationAngles;
    rotationAngles.assign(joint_num, 0);

    //rotationAxes = Local2global(rvec);
    rotationAxes = rvec;
    rotationAngles = normalize_vectors(rotationAxes);
    for (Point3D v : rotationAxes) {
        v.print();
    }
    /*********** local rodrigues로부터 global 좌표 추론 start **********/
    /*
        ====================================
        grand parent   - parent    - child
        ------------------------------------
        input          - base      - rotated
        ====================================
        inputPoints는 grand parent점의 좌표를 사용.
        base(parent)를 기준으로 한 상대적인 위치를 나타냄.
        
        skeletonPoints는 input을 rodrigues 벡터로 회전시킨 rotated 좌표를 저장.
        절대적인 위치를 나타내고, child점의 base가 됨. 
    */
    vector<Point3D> inputPoints;
    inputPoints.assign(joint_num, zeros);

    vector<Point3D> skeletonPoints;
    skeletonPoints.assign(joint_num, zeros);

    int mode = 2;
    Point3D false_pelvis(0.0, 0.0, 1.0);
    // 관절 사이 길이
    vector<float> limb_length = {1, 178.097, 142.38, 216.729
                                , 184.316, 142.705, 275.864, 231.535, 96.1261, 0, 0
                                , 183.71, 131.283, 281.083, 234.466,  103.501 ,0,0
                                , 91.024, 399.641, 381.934, 187.015
                                , 82.080, 399.211, 385.952, 172.063, 81.545, 160.411};
    for (int j = 0; j < joint_num; ++j) {
        if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
        int grand_j = joints_hierarchy[joints_hierarchy[j]];
        int parent_j = joints_hierarchy[j];
        // 역계산을 위해 각도 (-)
        
        if (grand_j == 0) {
            // grand_j가 0이면 inputPoints가 정의가 안됨. 매번 새로 만듦.(parent의 부호 반전으로)
            Point3D tempInput = inputPoints[grand_j].vmin(inputPoints[parent_j]);
            
            // 로드리게스 벡터로 회전
            if(mode ==1)
                skeletonPoints[j] = RodriguesRotation(tempInput, rotationAxes[j], -rotationAngles[j]);
            else
                skeletonPoints[j] = RodriguesRotation2(tempInput, rotationAxes[j], -rotationAngles[j]);
            // 관절 사이 길이 조절
            normalize_point(skeletonPoints[j]);
            skeletonPoints[j].mul(limb_length[j]);
            // global(절대좌표)로 바꾸기 위해 base를 더해줌.
            skeletonPoints[j].addv(skeletonPoints[joints_hierarchy[j]]);
        }
        else if (parent_j == 0) {
            // 로드리게스 벡터로 회전
            if (mode == 1)
                skeletonPoints[j] = RodriguesRotation(false_pelvis, rotationAxes[j], -rotationAngles[j]);
            else
                skeletonPoints[j] = RodriguesRotation2(false_pelvis, rotationAxes[j], -rotationAngles[j]);
            // 관절 사이 길이 조절
            normalize_point(skeletonPoints[j]);
            skeletonPoints[j].mul(limb_length[j]);
            // global(절대좌표)로 바꾸기 위해 base를 더해줌.
            skeletonPoints[j].addv(zeros);
        }
        else {
            // 로드리게스 벡터로 회전
            if (mode == 1)
                skeletonPoints[j] = RodriguesRotation(inputPoints[grand_j], rotationAxes[j], -rotationAngles[j]);
            else
                skeletonPoints[j] = RodriguesRotation2(inputPoints[grand_j], rotationAxes[j], -rotationAngles[j]);
            // 관절 사이 길이 조절
            normalize_point(skeletonPoints[j]);
            skeletonPoints[j].mul(limb_length[j]);
            // global(절대좌표)로 바꾸기 위해 base를 더해줌.
            skeletonPoints[j].addv(skeletonPoints[joints_hierarchy[j]]);
            
        }

        // local(상대좌표)로 input을 저장.
        inputPoints[j] = skeletonPoints[parent_j].vmin(skeletonPoints[j]);
        
    }
    /*********** local rodrigues로부터 global 좌표 추론 end **********/
    
    // 시각화(점, 선 그리기)
    vector<Point3D> pointCloud;
    visualizePoints(skeletonPoints, pointCloud);
    
    int point_count = pointCloud.size();
    cout << point_count << '\n';
    // 결과를 .ply 파일로 저장
    std::ofstream outFile("output_point_cloud.ply");
    if (outFile.is_open()) {
        outFile << "ply\n";
        outFile << "format ascii 1.0\n";
        outFile << "element vertex ";
        outFile << point_count << '\n';
        outFile << "property float x\n";
        outFile << "property float y\n";
        outFile << "property float z\n";
        outFile << "end_header\n";
        
        for (Point3D p : pointCloud) {
            outFile << p.x << " " << p.y << " " << p.z << "\n";
        }
        outFile.close();
        std::cout << "결과가 'output_point_cloud.ply' 파일에 저장되었습니다.\n";
    }
    else {
        std::cerr << "파일을 열 수 없습니다.\n";
        return 1;
    }
    
    return 0;
}

vector<vector<vector<vector<double>>>> load_ans_vector() {
    int small_motion_num = 2;
    const char* instruction[MOTION_NUM] = { "dummy.wav" , "기본준비서기A"};
    int motion_frames_limit[MOTION_NUM] = { 0 ,30,40 ,40 ,40 ,40 ,40,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40, 40 }; // 18동작
    int motion_frames_limit_ans[MOTION_NUM] = { 0,30, 73, 70, 70, 60, 80 , 60 , 70 , 90 , 60 , 80 , 80 , 100 , 80 , 100 ,80 , 60 ,320 }; //18동작
    int max_num = 80;
    
    vector<vector<vector<vector<double>>>> total_ans_vector;
    total_ans_vector.assign(small_motion_num,vector<vector<vector<double>>>(max_num, vector < vector<double>>(joint_num, vector<double>(3, 0))));

    int motion_index = 1;
    while (motion_index < small_motion_num) {

        vector < vector < vector<double>>> ans_vector;
        ans_vector.assign(motion_frames_limit_ans[motion_index], vector < vector<double>>(joint_num, vector<double>(3, 0)));

        fstream fs;
        string buf;
        string file_name2 = instruction[motion_index];
        file_name2 += ".csv";
        fs.open(file_name2, ios::in);
        int cnt = 0;
        float x = 0, y = 0, z = 0;
        
        //정답 가져오기
        while (cnt < motion_frames_limit_ans[motion_index]*joint_num*3) {

            if (cnt % 3 == 0) {
                getline(fs, buf, ',');
                x = stod(buf);
                ans_vector[(cnt / 3 / joint_num) % motion_frames_limit_ans[motion_index]][(cnt / 3) % joint_num][0] = x;
            }
            else if (cnt % 3 == 1) {
                getline(fs, buf, ',');
                y = stod(buf);
                ans_vector[(cnt / 3 / joint_num) % motion_frames_limit_ans[motion_index]][(cnt / 3) % joint_num][1] = y;
            }
            else if (cnt % 3 == 2) {
                getline(fs, buf, '\n');
                z = stod(buf);
                ans_vector[(cnt / 3 / joint_num) % motion_frames_limit_ans[motion_index]][(cnt / 3) % joint_num][2] = z;                                        // for test dtw
            }
            cnt++;


        }
        fs.close();

        total_ans_vector[motion_index] = ans_vector;
        ++motion_index;

    }

    return total_ans_vector;
}

vector<double> normalize_vectors(vector<Point3D>& v)
{
    
    int len = v.size();
    vector<double> a;
    a.assign(len, 0);
    for (int i = 0; i < len; ++i) {
        float div = sqrt(pow(v[i].x, 2) + pow(v[i].y, 2) + pow(v[i].z, 2));

        v[i].div(div);
        a[i]=div;
    }
    return a;
}
void normalize_point(Point3D& v) {
    float div = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
    v.div(div);
}

void  static Hierarchy_set() {
    joints_hierarchy.assign(28, 0);
    joints_hierarchy[0] = -1;
    joints_hierarchy[1] = 0, joints_hierarchy[18] = 0; joints_hierarchy[22] = 0; joints_hierarchy[2] = 1;//spine
    joints_hierarchy[3] = 2; joints_hierarchy[26] = 3; joints_hierarchy[27] = 26; //neck
    joints_hierarchy[19] = 18; joints_hierarchy[20] = 19; joints_hierarchy[21] = 20;  //left leg
    joints_hierarchy[23] = 22; joints_hierarchy[24] = 23; joints_hierarchy[25] = 24;  //right leg
    joints_hierarchy[4] = 2; joints_hierarchy[5] = 4; joints_hierarchy[6] = 5; joints_hierarchy[7] = 6;  joints_hierarchy[8] = 7;// left arm
    joints_hierarchy[11] = 2; joints_hierarchy[12] = 11; joints_hierarchy[13] = 12; joints_hierarchy[14] = 13; joints_hierarchy[15] = 14; // right arm
}