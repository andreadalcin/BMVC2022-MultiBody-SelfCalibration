function R = rotYPR(yaw, pitch, roll)
q = quaternion([yaw pitch roll],"eulerd","zyx","frame");
R = rotmat(q,"frame");
end