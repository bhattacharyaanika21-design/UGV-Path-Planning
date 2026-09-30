function scenario = replay_scenario(name)
%REPLAY_SCENARIO Replay an exported Python run inside MATLAB's drivingScenario.
%   scenario = replay_scenario("S1_village")
%
%   Requires Automated Driving Toolbox. Reads matlab/data/<name>/road.json,
%   actors.csv and ego.csv (created by `python3 export_matlab.py`), builds the
%   road network with road(), creates one actor per road user and steps the
%   scenario frame by frame so you can view it with plot()/chasePlot(), feed it
%   to drivingRadarDataGenerator / visionDetectionGenerator / lidarPointCloud-
%   Generator, or save it for Driving Scenario Designer (drivingScenarioDesigner(scenario)).

if nargin < 1, name = "S1_village"; end
d = fullfile(fileparts(mfilename("fullpath")), "data", name);
road_ = jsondecode(fileread(fullfile(d, "road.json")));
A = readtable(fullfile(d, "actors.csv"), "TextType", "string");
E = readtable(fullfile(d, "ego.csv"), "TextType", "string");

scenario = drivingScenario("SampleTime", 0.1, "StopTime", E.t(end));

% ---- roads (one road() per graph edge, no lane markings: Indian unstructured)
for k = 1:numel(road_.edges)
    e = road_.edges(k);
    pts = e.pts;
    if iscell(pts), pts = cell2mat(pts); end
    if size(pts, 1) < 2, continue; end
    road(scenario, [pts, zeros(size(pts, 1), 1)], e.width);
end

% ---- ego
egoV = vehicle(scenario, "ClassID", 1, "Length", 4.4, "Width", 1.8, ...
    "Position", [E.x(1) E.y(1) 0], "Yaw", rad2deg(E.yaw(1)), "PlotColor", [0 0.7 0.7]);

% ---- other road users
classMap = containers.Map( ...
    {'car','bus','truck','tractor','auto','twowheeler','bicycle','pedestrian','cattle','pushcart','bullockcart'}, ...
    {1, 2, 2, 2, 1, 3, 3, 4, 5, 5, 5});
ids = unique(A.id);
actorsById = containers.Map('KeyType', 'double', 'ValueType', 'any');
for i = 1:numel(ids)
    rows = A(A.id == ids(i), :);
    cls = char(rows.class(1));
    if classMap.isKey(cls), cid = classMap(cls); else, cid = 1; end
    ac = actor(scenario, "ClassID", cid, "Length", rows.length(1), "Width", rows.width(1), ...
        "Height", 1.5, "Position", [rows.x(1) rows.y(1) 0], "Yaw", rad2deg(rows.yaw(1)), ...
        "Name", string(cls) + "_" + ids(i));
    actorsById(ids(i)) = struct("actor", ac, "rows", rows);
end

% ---- playback
figure("Name", name); plot(scenario, "Waypoints", "off", "RoadCenters", "off");
title(road_.title);
tAll = E.t;
for k = 1:numel(tAll)
    t = tAll(k);
    egoV.Position = [E.x(k) E.y(k) 0];
    egoV.Yaw = rad2deg(E.yaw(k));
    egoV.Velocity = [E.v(k) * cos(E.yaw(k)), E.v(k) * sin(E.yaw(k)), 0];
    for i = 1:numel(ids)
        s = actorsById(ids(i));
        [dt, j] = min(abs(s.rows.t - t));
        if dt > 0.06   % actor not present at this time -> park it far away
            s.actor.Position = [1e4 + ids(i) * 10, 1e4, 0];
            continue
        end
        s.actor.Position = [s.rows.x(j) s.rows.y(j) 0];
        s.actor.Yaw = rad2deg(s.rows.yaw(j));
        s.actor.Velocity = [s.rows.v(j) * cos(s.rows.yaw(j)), s.rows.v(j) * sin(s.rows.yaw(j)), 0];
    end
    updatePlots(scenario);
    drawnow limitrate;
end
end
