function f = getInitialEstimateFromF0(F, f0, w, h)

K = [f0, 0,  w / 2; 
     0,  f0, h / 2; 
     0,  0,  1];

G = K' * F * K;
G = G / norm(G, 'fro');

% SVD
[U,S,Vh] = svd(G);
Vh = Vh';
s0 = S(1,1);
s1 = S(2,2);

% Quadratic equation
p = [s0^2 * (1 - U(3,1)^2) * (1 - Vh(1,3)^2) - s1^2 * (1 - U(3,2)^2) * (1 - Vh(2,3)^2);
    s0^2 * (U(3,1)^2 + Vh(1,3)^2 - 2 * U(3,1)^2 * Vh(1,3)^2) - s1^2 * (U(3,2)^2 + Vh(2,3)^2 - 2 * U(3,2)^2 * Vh(2,3)^2);
    s0^2 * U(3,1)^2 * Vh(1,3)^2 - s1^2 * U(3,2)^2 * Vh(2,3)^2];

rs = roots(p);
rs = real(rs(abs(imag(rs)) < 1e-6));
rs = rs(rs > 0);

if size(rs,1) > 0
    f = f0 * sqrt(rs(1));
else
    f = [];
end

end