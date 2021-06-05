/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles

  std::default_random_engine random_engine;
  
  std::normal_distribution<double> x_distribution(x, std[0]);
  std::normal_distribution<double> y_distribution(y, std[1]);
  std::normal_distribution<double> theta_distribution(theta, std[2]);
  
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    
    particle.x = x_distribution(random_engine);
    particle.y = y_distribution(random_engine);
    particle.theta = theta_distribution(random_engine);
    particle.weight = 1;
    
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine random_engine;

  std::normal_distribution<double> x_distribution(0, std_pos[0]);
  std::normal_distribution<double> y_distribution(0, std_pos[1]);
  std::normal_distribution<double> theta_distribution(0, std_pos[2]);

  double delta_theta = yaw_rate * delta_t;
  double eps = 1e-5;

  for (int i = 0; i < num_particles; ++i) {
    if ( fabs(yaw_rate) > eps ) {
      particles[i].x += velocity * ( sin( particles[i].theta + delta_theta) - sin(particles[i].theta ) ) / yaw_rate;
      particles[i].y += velocity * ( cos( particles[i].theta) - cos(particles[i].theta + delta_theta ) ) / yaw_rate;
      particles[i].theta += delta_theta;
    }
    else {
      particles[i].x += velocity * cos( particles[i].theta ) * delta_t;
      particles[i].y += velocity * sin( particles[i].theta ) * delta_t;
    }

    particles[i].x += x_distribution(random_engine);
    particles[i].y += y_distribution(random_engine);
    particles[i].theta += theta_distribution(random_engine);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

inline double multivariate2DGaussian(double x, double ux, double sigmaX, double y, double uy, double sigmaY){
  double gaussNorm = 1/ ( 2 * M_PI * sigmaX * sigmaY );
  double exponent = (x - ux) * (x - ux) / ( 2 * sigmaX * sigmaX ) + (y - uy) * (y - uy) / ( 2 * sigmaY * sigmaY );
  double prob =  gaussNorm * exp(-exponent) ;

  return prob;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; ++i) {

    double weight = 1.0;

    for (unsigned int j = 0; j < observations.size(); ++j) {
      LandmarkObs l;
      l.x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
      l.y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);

      double min_distance = 2 * sensor_range;
      double min_id = -1;

      for(unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
        double distance = dist(l.x, l.y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);

        if( distance <= sensor_range && distance < min_distance) {
          min_distance = distance;
          min_id = k;
        }
      }

      if( min_id > -1 ){
        double prob = multivariate2DGaussian(l.x, map_landmarks.landmark_list[min_id].x_f, std_landmark[0],
                          l.y, map_landmarks.landmark_list[min_id].y_f, std_landmark[1]);
        weight *= prob;
      }

    }

    particles[i].weight = weight;
    weights[i] = weight;

  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> discrete_dist(weights.begin(), weights.end());

  vector<Particle> new_particles(num_particles);
  for (int i = 0; i < num_particles; ++i) {

    int particle_id = discrete_dist(gen);

    new_particles[i].x = particles[particle_id].x;
    new_particles[i].y = particles[particle_id].y;
    new_particles[i].theta = particles[particle_id].theta;
    new_particles[i].weight = particles[particle_id].weight;

    weights[i] = particles[particle_id].weight;
  }

  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}