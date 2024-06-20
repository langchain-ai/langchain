/* eslint-disable no-return-assign, react/jsx-props-no-spreading */
import React, { useState, useEffect } from "react";
import { createClient } from "@supabase/supabase-js";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { v4 as uuidv4 } from "uuid";

const useCookie = () => {
  /**
   * Function to set a cookie
   * @param {string} name The name of the cookie to set
   * @param {string} value The value of the cookie
   * @param {number} days the number of days until the cookie expires
   */
  const setCookie = (name, value, days) => {
    const d = new Date();
    d.setTime(d.getTime() + days * 24 * 60 * 60 * 1000);
    const expires = `expires=${d.toUTCString()}`;
    document.cookie = `${name}=${value};${expires};path=/`;
  };

  /**
   * Function to get a cookie by name
   * @param {string} name The name of the cookie to get
   * @returns {string} The value of the cookie
   */
  const getCookie = (name) => {
    const ca = document.cookie.split(";");
    const caLen = ca.length;
    const cookieName = `${name}=`;
    let c;

    for (let i = 0; i < caLen; i += 1) {
      c = ca[i].replace(/^\s+/g, "");
      if (c.indexOf(cookieName) === 0) {
        return c.substring(cookieName.length, c.length);
      }
    }
    return "";
  };

  /**
   * Function to check cookie existence
   * @param {string} name The name of the cookie to check for
   * @returns {boolean} Whether or not the cookie exists
   */
  const checkCookie = (name) => {
    const cookie = getCookie(name);
    if (cookie !== "") {
      return true;
    }
    return false;
  };

  return { setCookie, checkCookie };
};

function SvgThumbsUp() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth="1.5"
      stroke="#166534"
      style={{ width: "24px", height: "24px" }}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M6.633 10.25c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 0 1 2.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 0 0 .322-1.672V2.75a.75.75 0 0 1 .75-.75 2.25 2.25 0 0 1 2.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282m0 0h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 0 1-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 0 0-1.423-.23H5.904m10.598-9.75H14.25M5.904 18.5c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 0 1-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 9.953 4.167 9.5 5 9.5h1.053c.472 0 .745.556.5.96a8.958 8.958 0 0 0-1.302 4.665c0 1.194.232 2.333.654 3.375Z"
      />
    </svg>
  );
}

function SvgThumbsDown() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth="1.5"
      stroke="#991b1b"
      style={{ width: "24px", height: "24px" }}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M7.498 15.25H4.372c-1.026 0-1.945-.694-2.054-1.715a12.137 12.137 0 0 1-.068-1.285c0-2.848.992-5.464 2.649-7.521C5.287 4.247 5.886 4 6.504 4h4.016a4.5 4.5 0 0 1 1.423.23l3.114 1.04a4.5 4.5 0 0 0 1.423.23h1.294M7.498 15.25c.618 0 .991.724.725 1.282A7.471 7.471 0 0 0 7.5 19.75 2.25 2.25 0 0 0 9.75 22a.75.75 0 0 0 .75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 0 0 2.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384m-10.253 1.5H9.7m8.075-9.75c.01.05.027.1.05.148.593 1.2.925 2.55.925 3.977 0 1.487-.36 2.89-.999 4.125m.023-8.25c-.076-.365.183-.75.575-.75h.908c.889 0 1.713.518 1.972 1.368.339 1.11.521 2.287.521 3.507 0 1.553-.295 3.036-.831 4.398-.306.774-1.086 1.227-1.918 1.227h-1.053c-.472 0-.745-.556-.5-.96a8.95 8.95 0 0 0 .303-.54"
      />
    </svg>
  );
}

/**
 * Generated type for the Supabase DB schema.
 * @typedef {import('../supabase').Database} Database
 */

const FEEDBACK_COOKIE_PREFIX = "feedbackSent";
/** @type {Database["public"]["Enums"]["project_type"]} */
const LANGCHAIN_PROJECT_NAME = "langchain_py_docs";

/**
 * @returns {Promise<string>}
 */
const getIpAddress = async () => {
  const response = await fetch("https://api.ipify.org?format=json");
  return (await response.json()).ip;
};

export default function Feedback() {
  const { setCookie, checkCookie } = useCookie();
  const [feedbackId, setFeedbackId] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackDetailsSent, setFeedbackDetailsSent] = useState(false);
  const { siteConfig } = useDocusaurusContext();
  const [pathname, setPathname] = useState("");

  /** @param {"good" | "bad"} feedback */
  const handleFeedback = async (feedback) => {
    if (process.env.NODE_ENV !== "production") {
      console.log("Feedback (dev)");
      return;
    }

    const cookieName = `${FEEDBACK_COOKIE_PREFIX}_${window.location.pathname}`;
    if (checkCookie(cookieName)) {
      return;
    }

    /** @type {Database} */
    const supabase = createClient(
      siteConfig.customFields.supabaseUrl,
      siteConfig.customFields.supabasePublicKey
    );
    try {
      const ipAddress = await getIpAddress();
      const rowId = uuidv4();
      setFeedbackId(rowId);
      const params = {
        id: rowId,
        is_good: feedback === "good",
        url: window.location.pathname,
        user_ip: ipAddress,
        project: LANGCHAIN_PROJECT_NAME,
      };

      const { error } = await supabase.from("feedback").insert(params);
      if (error) {
        throw error;
      }
    } catch (e) {
      console.error("Failed to send feedback", e);
      return;
    }

    // Set a cookie to prevent feedback from being sent multiple times
    setCookie(cookieName, window.location.pathname, 1);
    setFeedbackSent(true);
  };

  const handleFeedbackDetails = async (e) => {
    e.preventDefault();
    if (!feedbackId) {
      setFeedbackDetailsSent(true);
      return;
    }
    const details = e.target.elements
      .namedItem("details")
      ?.value.slice(0, 1024);
    if (!details) {
      return;
    }
    const supabase = createClient(
      siteConfig.customFields.supabaseUrl,
      siteConfig.customFields.supabasePublicKey
    );
    const { error } = await supabase.from("feedback_details").insert({
      feedback_id: feedbackId,
      details,
    });
    if (error) {
      console.error("Failed to add feedback details", error);
      return;
    }
    setFeedbackDetailsSent(true);
  };

  useEffect(() => {
    if (typeof window !== "undefined") {
      // If the cookie exists, set feedback sent to
      // true so the user can not send feedback again
      // (cookies exp in 24hrs)
      const cookieName = `${FEEDBACK_COOKIE_PREFIX}_${window.location.pathname}`;
      setFeedbackSent(checkCookie(cookieName));
      setPathname(window.location.pathname);
    }
  }, []);

  const defaultFields = {
    style: {
      display: "flex",
      alignItems: "center",
      paddingTop: "10px",
      paddingBottom: "10px",
      paddingLeft: "22px",
      paddingRight: "22px",
      border: "1px solid gray",
      borderRadius: "6px",
      gap: "10px",
      cursor: "pointer",
      fontSize: "16px",
      fontWeight: "600",
    },
    onMouseEnter: (e) => (e.currentTarget.style.backgroundColor = "#f0f0f0"),
    onMouseLeave: (e) =>
      (e.currentTarget.style.backgroundColor = "transparent"),
    onMouseDown: (e) => (e.currentTarget.style.backgroundColor = "#d0d0d0"),
    onMouseUp: (e) => (e.currentTarget.style.backgroundColor = "#f0f0f0"),
  };

  const newGithubIssueURL = pathname
    ? `https://github.com/langchain-ai/langchain/issues/new?assignees=&labels=03+-+Documentation&projects=&template=documentation.yml&title=DOC%3A+%3CIssue+related+to+${pathname}%3E&url=https://python.langchain.com${pathname}`
    : "https://github.com/langchain-ai/langchain/issues/new?assignees=&labels=03+-+Documentation&projects=&template=documentation.yml&title=DOC%3A+%3CPlease+write+a+comprehensive+title+after+the+%27DOC%3A+%27+prefix%3E";

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <hr />
      {feedbackSent ? (
        <>
          <h4>Thanks for your feedback!</h4>
          {!feedbackDetailsSent && feedbackId && (
            <form
              style={{ display: "flex", flexDirection: "column" }}
              onSubmit={handleFeedbackDetails}
            >
              <h4>Do you have any specific comments?</h4>
              <textarea
                name="details"
                style={{ width: "480px", height: "120px" }}
              />
              <button
                style={{
                  width: "72px",
                  marginLeft: "408px",
                  marginTop: "12px",
                }}
                type="submit"
              >
                Submit
              </button>
            </form>
          )}
        </>
      ) : (
        <>
          <h4>Was this page helpful?</h4>
          <div style={{ display: "flex", gap: "5px" }}>
            <div
              {...defaultFields}
              role="button" // Make it recognized as an interactive element
              tabIndex={0} // Make it focusable
              onKeyDown={async (e) => {
                // Handle keyboard interaction
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  await handleFeedback("good");
                }
              }}
              onClick={async (e) => {
                e.preventDefault();
                await handleFeedback("good");
              }}
            >
              <SvgThumbsUp />
            </div>
            <div
              {...defaultFields}
              role="button" // Make it recognized as an interactive element
              tabIndex={0} // Make it focusable
              onKeyDown={async (e) => {
                // Handle keyboard interaction
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  await handleFeedback("bad");
                }
              }}
              onClick={async (e) => {
                e.preventDefault();
                await handleFeedback("bad");
              }}
            >
              <SvgThumbsDown />
            </div>
          </div>
        </>
      )}
      <br />
      <h4>
        You can also leave detailed feedback{" "}
        <a target="_blank" href={newGithubIssueURL}>
          on GitHub
        </a>
        .
      </h4>
    </div>
  );
}
