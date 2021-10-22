clc
global std_x std_z C x0 T a h A z x_all

std_x = 0.02;
std_z = 0.01;
C = 0.05;
x0 = 0.1;
T = 200;
MC = 100;
a = @(xk,noise) -0.05 + sin(xk) + C*xk + noise*std_x*randn(numel(xk),1);
h = @(xk,noise) abs(xk) + noise*std_z*randn(numel(xk),1);
A = @(xk) cos(xk) + C;

x_all = zeros(1,T);
x_all_n = zeros(1,T);
z = zeros(1,T);
x_all(1) = x0;
x_all_n(1) = x0;

for i = 2:T
    x_all(i) = a(x_all(i-1),0);
    z(i) = h(x_all(i),1);
    x_all_n(i) = a(x_all(i-1),1);
end

plot(1:T,x_all)
hold on
plot(1:T,x_all_n)

errors = zeros(5,MC);
for i=1:MC
    errors(1,i) = ekf_explicit();
    errors(2,i) = ekf_implicit();
    errors(3,i) = dhf_explicit(0);
    errors(4,i) = dhf_explicit(1);
    errors(5,i) = dhf_implicit();
end
mean(errors,2)

function error = ekf_explicit()
    global std_x std_z x0 T a h A z x_all
    x_filt = zeros(1,T);
    x_filt(1) = x0;
    cov_x = zeros(1,T);
    cov_x(1) = 0.001;

    for i = 2:T
        x_pred = a(x_filt(i-1),0);
        cov_pred = A(x_filt(i-1))*cov_x(i-1)*A(x_filt(i-1)) + std_x^2;

        K = cov_pred*H(x_pred)/(H(x_pred)*cov_pred*H(x_pred) + std_z^2);
        x_filt(i) = x_pred + K*(z(i)-h(x_pred,0));
        cov_x(i) = (1-K*H(x_pred))*cov_pred;
    end
    plot(1:T,x_filt)
    error = sqrt(mean((x_filt-x_all).^2));
end

function error = ekf_implicit()
    global std_x std_z x0 T a A z x_all
   
    Hz = @(zk) -2*zk;
    Hx = @(xk) 2*xk;
    h_impl = @(xk, zk) xk^2-zk^2;
    x_filt = zeros(1,T);
    x_filt(1) = x0;
    cov_x = zeros(1,T);
    cov_x(1) = 0.001;

    for i = 2:T
        x_pred = a(x_filt(i-1),0);
        cov_pred = A(x_filt(i-1))*cov_x(i-1)*A(x_filt(i-1)) + std_x^2;

        K = cov_pred*Hx(x_pred)/(Hx(x_pred)*cov_pred*Hx(x_pred) + Hz(z(i))*std_z^2*Hz(z(i)));
        x_filt(i) = x_pred + K*(-h_impl(x_pred,z(i)));
        cov_x(i) = (1-K*H(x_pred))*cov_pred;
    end
    plot(1:T,x_filt)
    error = sqrt(mean((x_filt-x_all).^2));
end

function error = dhf_explicit(corrigated)
    global std_x std_z x0 T a h A z x_all
    
    p = 500;
    x_particles = zeros(p,T);
    x_particles(:,1) = mvnrnd(x0,0.1,p);
    d_lambda = 0.1;
    x_avg_ekf = mean(x_particles(:,1));
    cov_ekf = 0.001;
    
    x_filt = zeros(1,T);
    x_filt(1) = x_avg_ekf;

    for i = 2:T
        x_particles(:,i) = a(x_particles(:,i-1),1);
        x_filt(i) = mean(x_particles(:,i));
        
        x_avg_pred_ekf = a(x_avg_ekf,0);
        cov_pred_ekf = A(x_avg_ekf)*cov_ekf*A(x_avg_ekf) + std_x^2;

        for j = 1:1/d_lambda
            lambda = j*d_lambda;
            H_lin = H(x_filt(i));
            gamma = h(x_filt(i),0)-H_lin*x_filt(i);
            B = -0.5*cov_pred_ekf*H_lin/(lambda*H_lin*cov_pred_ekf*H_lin + std_z^2)*H_lin;
            b = (1+2*lambda*B)*((1+lambda*B)*cov_pred_ekf*H_lin/std_z^2*(z(i)-corrigated*gamma) + B*x_filt(i));
            x_particles(:,i) = x_particles(:,i) + d_lambda*(B*x_particles(:,i)+b);
            x_filt(i) = mean(x_particles(:,i));
        end
        
        H_lin = H(x_avg_pred_ekf);
        K = cov_pred_ekf*H_lin/(H_lin*cov_pred_ekf*H_lin + std_z^2);
        x_avg_ekf = x_avg_pred_ekf + K*(z(i)-h(x_avg_pred_ekf,0));
        cov_ekf = (1-K*H_lin)*cov_pred_ekf;
    end
    plot(1:T,x_filt)
    error = sqrt(mean((x_filt-x_all).^2));
end

function error = dhf_implicit()
    global std_x std_z x0 T a h A z x_all
    
    Hz = @(zk) 2*zk;
    Hx = @(xk) 2*xk;
    h_impl = @(xk, zk) xk^2-zk^2;
    p = 500;
    
    x_particles = zeros(p,T);
    x_particles(:,1) = mvnrnd(x0,0.1,p);
    d_lambda = 0.1;
    x_avg_ekf = mean(x_particles(:,1));
    cov_ekf = 0.001;
    
    x_filt = zeros(1,T);
    x_filt(1) = x_avg_ekf;

    for i = 2:T
        x_particles(:,i) = a(x_particles(:,i-1),1);
        x_filt(i) = mean(x_particles(:,i));
        
        x_avg_pred_ekf = a(x_avg_ekf,0);
        cov_pred_ekf = A(x_avg_ekf)*cov_ekf*A(x_avg_ekf) + std_x^2;
        
        H_lin_z = Hz(z(i));
        for j = 1:1/d_lambda
            lambda = j*d_lambda;
            H_lin = Hx(x_filt(i));
            y = -h_impl(x_filt(i),z(i))+H_lin*x_filt(i);
            B = -0.5*cov_pred_ekf*H_lin/(lambda*H_lin*cov_pred_ekf*H_lin + H_lin_z*std_z^2*H_lin_z)*H_lin;
            b = (1+2*lambda*B)*((1+lambda*B)* ...
                cov_pred_ekf*H_lin/(H_lin_z*std_z^2*H_lin_z)* ...
                y + B*x_filt(i));
            x_particles(:,i) = x_particles(:,i) + d_lambda*(B*x_particles(:,i)+b);
            x_filt(i) = mean(x_particles(:,i));
        end 
        
        H_lin = Hx(x_avg_pred_ekf);
        K = cov_pred_ekf*H_lin/(H_lin*cov_pred_ekf*H_lin + std_z^2);
        x_avg_ekf = x_avg_pred_ekf + K*(z(i)-h(x_avg_pred_ekf,0));
        cov_ekf = (1-K*H_lin)*cov_pred_ekf;
    end
    plot(1:T,x_filt)
    error = sqrt(mean((x_filt-x_all).^2));
end

function out = H(xk)
    if (xk < 0)
        out = -1;
    elseif (xk > 0)
        out = 1;
    else
        disp("x cannot be 0")
        out = NaN;
    end
end