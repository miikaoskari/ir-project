% Load data
data = readtable('data/df_total.csv');

enable_mahal = false;

% Extract data
timestamps = data.timestamps / 1000; % Convert to seconds
true_x = data.true_x; true_y = data.true_y;
true_lat = data.true_lat; true_lon = data.true_lon;
ax = data.ax; ay = data.ay;
gnss_x = data.gnss_x; gnss_y = data.gnss_y;
gnss_lat = data.gnss_lat; gnss_lon = data.gnss_lon;

% Basic EKF Configuration
dt = mean(diff(timestamps)); % Average time step
state = [gnss_x(1); gnss_y(1); 0; 0]; % [x; y; vx; vy]
P = diag([10, 10, 1, 1]); % Initial covariance

Q = diag([0.1, 0.1, 0.5, 0.5]); % Process noise (position and velocity)
R = diag([3^2, 3^2]); % Measurement noise (GNSS)

% Preallocate results
num_steps = numel(timestamps);
estimated_states = zeros(num_steps, 4);

ay = ay - 9.81;  % Remove gravitational bias

% Main loop
for i = 1:num_steps
    % Prediction (Constant acceleration model)
    A = [1 0 dt 0;  % State transition matrix
         0 1 0 dt;
         0 0 1 0;
         0 0 0 1];
    
    B = [0.5*dt^2 0;  % Control matrix
          0 0.5*dt^2;
          dt 0;
          0 dt];
    
    state = A * state + B * [ax(i); ay(i)];
    P = A * P * A' + Q;
    
    % Correction (GNSS update)
    if ~isnan(gnss_x(i)) && ~isnan(gnss_y(i))
        H = [1 0 0 0;  % Measurement matrix
             0 1 0 0];
        
        y = [gnss_x(i); gnss_y(i)] - H * state; % innovation residual
        S = H * P * H' + R; % innovation covariance

        if enable_mahal
            d2 = y' * (S \ y); % mahalanobis distance
            threshold = 9.21; % chi-squared threshold 

            if d2 >= threshold
                fprintf('Measurement rejected at pos (%d, %d)\n', i, d2);
                % hop to next iteration
                continue;
            end
        end

        K = P * H' / S; % kalman gain
        state = state + K * y; % update state estimate
        P = (eye(4) - K * H) * P;

    end
    
    estimated_states(i,:) = state';
end

% Calculate position errors
position_errors = sqrt( (estimated_states(:,1) - true_x).^2 + ...
                       (estimated_states(:,2) - true_y).^2 );

% Calculate error statistics
rmse = sqrt(mean(position_errors.^2));
mean_error = mean(position_errors);
max_error = max(position_errors);

% Print performance metrics
fprintf('--- Error Metrics ---\n');
fprintf('RMSE: %.2f meters\n', rmse);
fprintf('Mean Error: %.2f meters\n', mean_error);
fprintf('Max Error: %.2f meters\n\n', max_error);

% Plot error analysis
figure;
subplot(2,1,1);
plot(timestamps, position_errors);
title('Position Error Over Time');
xlabel('Time (s)');
ylabel('Error (m)');
grid on;

subplot(2,1,2);
histogram(position_errors, 20);
title('Error Distribution');
xlabel('Error (m)');
ylabel('Frequency');
grid on;

figure;
plot(true_x, true_y, 'k-', 'LineWidth', 1.5);
hold on;
plot(estimated_states(:,1), estimated_states(:,2), 'b--');
plot(gnss_x, gnss_y, 'r.', 'MarkerSize', 8);
hold off;
legend('Ground Truth', 'EKF Estimate', 'GNSS Measurements', ...
       'Location', 'best');
title(sprintf('Position Tracking (RMSE: %.2f m)', rmse));
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;